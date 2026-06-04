# Modelling Refactor: Per-Municipio Models

**Status:** Design / handoff document  
**Context:** `make municipio-day-train` baseline reached R²≈0.30 (train/val/test consistent, no
overfitting). This document analyses why the global model is fundamentally limited and lays out
the architectural choices for moving to per-municipio models.

---

## 1. Current Architecture (Baseline)

```
featurize phase
───────────────
gold panel (117 110 rows, 35 municipios)
  → enriched panel + feature registry (parquet + json)

train phase
───────────
enriched panel
  → _build_municipio_sequences (per municipio, loop)
      ├─ channel 0: accident_count z-scored with training-period stats
      └─ channels 1-9: cyclical calendar + holiday-proximity features
  → _assemble_split_arrays
      ├─ X_seq: (86 310, 10, 90)  all municipalities, combined
      ├─ y_scaled: z-scored targets
      ├─ y_raw / y_mean / y_std: inverse-transform bookkeeping
      └─ X_oh: (86 310, 35) municipality one-hot columns
  → MiniRocket.fit(X_seq_train)          # global kernel set
  → MiniRocket.transform(X_seq)          # 10 000 PPV features per sample
  → X_full = [X_rocket | X_oh]           # (86 310, 10 031)
  → TruncatedSVD(300).fit_transform(X_full_train)   ← memory guard
  → RidgeCV(LOOCV) on (86 310, 300)
  → single Pipeline artifact
```

**Baseline metrics (train 2017-2023, val 2024, test 2025-2026):**

| split      | MAE    | RMSE   | R²     |
|------------|--------|--------|--------|
| train      | 0.3515 | 0.5893 | 0.3341 |
| validation | 0.3632 | 0.6037 | 0.3238 |
| test       | 0.3612 | 0.6003 | 0.3025 |

Units are accidents/day per municipio (inverse z-score applied at evaluation time).
Implied target σ ≈ 0.72 accidents/day; the model explains ~30 % of daily variance.

---

## 2. Why the Global Model Has a Hard Ceiling

### 2.1 One-hot encoding is only an intercept shift

The current design concatenates 35 one-hot columns to the 10 000 MiniRocket PPV features and
feeds the combined matrix to a single Ridge. In a linear model this is equivalent to fitting one
global weight vector **w** and allowing each municipio to have its own intercept:

```
ŷ = w_rocket · x_rocket + w_mun_i   (for municipio i)
```

The weight vector `w_rocket` is shared across all 35 municipios. The model cannot learn that:
- Florianópolis peaks on summer-tourism weekends while Joinville peaks on weekday commuter traffic.
- A municipio traversed by heavy freight has a different day-of-week pattern than a beach corridor.
- A small municipio has near-zero baseline and almost all signal is in the holiday spikes.

### 2.2 TruncatedSVD discards 97 % of the MiniRocket signal

The SVD was added as a memory guard for LOOCV on the (86 310, 10 031) matrix. With 300 retained
components the model only sees ~3 % of the MiniRocket feature variance. This is the single
largest source of avoidable information loss.

The mathematical bound: for a rank-300 approximation of an 86 310 × 10 031 matrix, the fraction
of explained variance depends on the spectrum of the data matrix. MiniRocket PPV features tend to
have a fairly flat spectrum (many moderately informative features), so truncating at 300 likely
discards far more useful signal than a comparable truncation would in a PCA of natural images.

### 2.3 Shared kernel set may be suboptimal

MiniRocket is fitted on the pooled training corpus. The random dilated convolution kernels are
data-independent (their parameters are fixed once the time-series length is set), but the bias
values (`_fit_biases`) are derived from the pooled empirical distribution. A municipio with
anomalous dynamics may be poorly served by biases that are diluted across 35 municipios.

---

## 3. Architectural Options

Three options are presented in increasing complexity and expected benefit.

---

### Option A — Shared MiniRocket, Per-Municipio Ridge (Recommended)

```
featurize phase (unchanged)
  → enriched panel + registry

train phase (changed)
  → featurize() builds per-municipio sequence arrays (no concatenation)
  → MiniRocket.fit(X_seq_train_all)      # global kernel set (unchanged)
  → for each municipio i:
      X_rocket_i = MiniRocket.transform(X_seq_train_i)   # (n_i, 10 000)
      RidgeCV_i(LOOCV) on (n_i, 10 000)                  # NO SVD
  → 35 Ridge models stored in one dict artifact
```

**Why this is the right balance:**
- MiniRocket is fitted on ~86 000 training sequences; more data → better bias estimation and PPV
  stability. No reason to discard cross-municipio information here.
- Ridge is fitted per-municipio; ~2 466 samples × 10 000 features. LOOCV U-matrix is
  2 466 × 2 466 × 8 bytes ≈ **49 MB** — completely feasible on any laptop. No SVD needed.
- Each Ridge model learns the full 10 000-dimensional MiniRocket manifold for its municipio.
- The one-hot columns are no longer needed (each model is inherently municipio-specific).

**Memory budget for Option A:**

| Step | Peak memory |
|------|-------------|
| MiniRocket fit: X_seq_all (86 310, 10, 90) float32 | ~310 MB |
| MiniRocket transform: can be done per-municipio sequentially | ~100 MB per pass |
| Ridge LOOCV per municipio: U matrix (2 466, 2 466) float64 | ~49 MB |
| 35 fitted Ridge objects in memory simultaneously | ~35 × ~8 MB weights = ~280 MB |

Total peak is well under 1 GB with sequential per-municipio transforms.

---

### Option B — Fully Independent Per-Municipio MiniRocket + Ridge

```
for each municipio i:
    MiniRocket_i.fit(X_seq_train_i)
    X_rocket_i = MiniRocket_i.transform(X_seq_i)
    RidgeCV_i.fit(X_rocket_i, y_i)
```

**Advantage:** Each MiniRocket sees only one municipio's distribution; biases perfectly calibrated
per series.  
**Disadvantage:** ~2 466 sequences per municipio is on the low end for stable MiniRocket bias
estimation (the paper recommends ≥1 000 but finds better results with more). The pooled fit in
Option A uses 35× more data for essentially the same computational cost since MiniRocket's kernel
parameters are independent of the data. Only the bias values change, and pooled biases are
generally more stable for small series.  
**Verdict:** Unlikely to outperform Option A; more complex artifact management (35 MiniRocket
transformers). Not recommended.

---

### Option C — Hierarchical / Mixed Global + Local

Keep the global model as a prior (for municipios with very few samples or new municipios) and
train per-municipio residual models. Requires more complex infrastructure and is premature given
the baseline is only at R²≈0.30. **Defer** until Options A/B are benchmarked.

---

## 4. TruncatedSVD: Keep, Remove, or Make Optional

### Decision matrix

| Scenario | SVD needed? | Recommendation |
|----------|-------------|----------------|
| Global model, LOOCV, n=86 310, p=10 031 | Yes — U matrix would be 6.9 GB | Keep at 300–1 000 |
| Global model, k-fold (cv=5), `solver='lsqr'` | No U matrix formed | Removable |
| Per-municipio (Option A), n≈2 466, p=10 000 | No — U matrix is 49 MB | **Remove** |

### Short-term cheap win (without refactoring to per-municipio)

Before committing to the full refactor, increase `svd_components` in `experiment.yaml`:

```yaml
svd_components: 1000   # was 300 — retains ~10 % instead of ~3 % of signal
```

This is a one-line config change that does not require refeaturization. Expected improvement:
moderate (addresses the 97 % truncation loss, but does not fix the shared-weights problem).

Alternatively, drop SVD and switch to k-fold + iterative solver:

```python
Pipeline([
    ("ridge", RidgeCV(
        alphas=cfg.ridge_alphas,
        scoring="neg_mean_squared_error",
        cv=5,
        solver="lsqr",   # iterative; does not form Gram matrix
    )),
])
```

`solver='lsqr'` still needs to materialise the full X matrix in memory (~6.9 GB) but avoids
the U-matrix computation. Whether this fits in memory depends on the machine.

### After the refactor to Option A

`TruncatedSVD` and the `sklearn.pipeline.Pipeline` wrapper become dead code and should be
removed. The `svd_components` config field and its env override
`ML_MUNICIPIO_DAY_SVD_COMPONENTS` can be deprecated.

---

## 5. Code Changes Required for Option A

### 5.1 `_assemble_split_arrays` — add per-municipio view

Currently returns a concatenated array over all municipios. Add a sibling function (or a
`per_municipio=True` flag) that returns a dict of arrays keyed by `codigo_municipio`:

```python
def _assemble_per_municipio_arrays(
    panel: pd.DataFrame,
    registry: dict[str, Any],
    lookback_days: int,
    split: str,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns {mun_id: {"X_seq": ..., "y_scaled": ..., "y_raw": ...,
                       "y_mean": float, "y_std": float}}
    """
```

The existing `_assemble_split_arrays` can be kept for the global MiniRocket fit (it produces the
combined training corpus).

### 5.2 `featurize()` — store per-municipio sequence arrays

```python
# After fitting MiniRocket on the global training corpus:
per_mun_seqs: dict[str, dict[str, Any]] = {}
for split_name in ("train", "validation", "test"):
    per_mun_seqs[split_name] = _assemble_per_municipio_arrays(
        panel, registry, lookback_days, split_name
    )
feature_data["per_mun_seqs"] = per_mun_seqs
```

The existing combined `split_data` key can be removed or kept for backwards compatibility.

### 5.3 `train()` — loop over municipios

```python
def train(self, feature_data):
    rocket = feature_data["rocket"]
    per_mun_seqs = feature_data["per_mun_seqs"]
    registry = feature_data["registry"]
    municipio_codes = registry["municipio_codes"]

    models: dict[str, RidgeCV] = {}
    alphas: dict[str, float] = {}

    for mun_id in municipio_codes:
        train_data = per_mun_seqs["train"].get(mun_id)
        if train_data is None or len(train_data["y_scaled"]) == 0:
            self.logger.warning("No training data for municipio=%s, skipping", mun_id)
            continue

        X_rocket = rocket.transform(train_data["X_seq"]).astype(np.float32)
        ridge = RidgeCV(alphas=cfg.ridge_alphas, scoring="neg_mean_squared_error", cv=None)
        ridge.fit(X_rocket, train_data["y_scaled"])

        models[mun_id] = ridge
        alphas[mun_id] = float(ridge.alpha_)
        self.logger.info("municipio=%-20s  alpha=%.4g  n_train=%d",
                         mun_id, ridge.alpha_, len(train_data["y_scaled"]))

    return {"models": models, "alphas": alphas}
```

Note: `TruncatedSVD` and `Pipeline` imports can be removed.

### 5.4 `evaluate()` — per-municipio predict and aggregate

```python
def evaluate(self, feature_data, training_output):
    rocket = feature_data["rocket"]
    models = training_output["models"]
    per_mun_seqs = feature_data["per_mun_seqs"]
    metrics = {}

    for split_name in ("train", "validation", "test"):
        y_true_all, y_pred_all = [], []
        per_mun_metrics = {}

        for mun_id, model in models.items():
            split_data = per_mun_seqs[split_name].get(mun_id)
            if split_data is None or len(split_data["y_raw"]) == 0:
                continue
            X_rocket = rocket.transform(split_data["X_seq"]).astype(np.float32)
            y_pred_scaled = model.predict(X_rocket)
            y_pred = y_pred_scaled * split_data["y_std"] + split_data["y_mean"]
            y_true = split_data["y_raw"]

            mae_m = float(mean_absolute_error(y_true, y_pred))
            rmse_m = float((mean_squared_error(y_true, y_pred)) ** 0.5)
            r2_m = float(r2_score(y_true, y_pred))
            per_mun_metrics[mun_id] = {"mae": mae_m, "rmse": rmse_m, "r2": r2_m}

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

        if y_true_all:
            y_true_cat = np.concatenate(y_true_all)
            y_pred_cat = np.concatenate(y_pred_all)
            metrics[split_name] = {
                "mae":  float(mean_absolute_error(y_true_cat, y_pred_cat)),
                "rmse": float(mean_squared_error(y_true_cat, y_pred_cat) ** 0.5),
                "r2":   float(r2_score(y_true_cat, y_pred_cat)),
                "per_municipio": per_mun_metrics,
            }

    return metrics
```

### 5.5 `persist()` — single dict artifact for all models

```python
# Replace:
joblib.dump(training_output["ridge"], str(ridge_local))

# With:
joblib.dump(training_output["models"], str(ridge_local))
# Artifact filename kept as ridge_model.joblib for datalake path continuity.
```

The dict `{mun_id: RidgeCV}` serialises cleanly with joblib. At inference, load the dict and
dispatch on `codigo_municipio`.

### 5.6 `mlflow_params()` — add per-municipio alpha summary

```python
"ridge_alpha_chosen": json.dumps(training_output["alphas"]),
# And remove svd_components
```

Consider logging per-municipio R² values as individual MLflow metrics:
```python
# In mlflow_metrics():
for mun_id, m in evaluation_output["test"].get("per_municipio", {}).items():
    flat[f"r2__test__{mun_id}"] = m["r2"]
```

### 5.7 Config changes

In `experiment.yaml`, `svd_components` can be removed (or kept with a deprecation comment).
The `MunicipioDayExperimentConfig` dataclass field and the `ML_MUNICIPIO_DAY_SVD_COMPONENTS` env
override should be cleaned up.

---

## 6. Artifact Layout After Refactor

| File | Contents |
|------|----------|
| `minirocket_transformer.joblib` | Single fitted `MiniRocket` instance (shared, unchanged) |
| `ridge_model.joblib` | `dict[str, RidgeCV]` — one fitted Ridge per municipio |
| `evaluation_metrics.parquet` | Aggregate metrics per split (as now) |
| `per_municipio_metrics.parquet` | Per-municipio MAE/RMSE/R² per split (new) |
| `feature_registry.json` | Municipio code list, split config, z-score params (unchanged) |

---

## 7. Memory and Runtime Estimates

| Step | Current (global) | Refactored (per-municipio) |
|------|-------------------|---------------------------|
| MiniRocket fit | 310 MB, ~180 s | 310 MB, ~180 s (unchanged) |
| MiniRocket transform (train) | 3.4 GB combined | ~100 MB × 35 sequential ≈ peak 100 MB |
| Ridge fit | 2 GB (SVD) + 200 MB | 35 × 49 MB sequential ≈ peak 150 MB |
| Total train wall-time | ~4 min featurize + ~1 min Ridge | ~4 min featurize + ~5 min Ridge (35×) |
| Artifact size | ~80 MB (1 Ridge) | ~280 MB (35 Ridge objects) |

Sequential transform-then-fit per municipio keeps peak RAM well under 1 GB.

---

## 8. Open Questions

1. **Municipios with very few training samples.** If a municipio has `n < 200` training sequences
   (possible for very small or recently-added municipios), LOOCV Ridge may be unstable. A
   per-municipio sample-count floor should trigger a warning or a fallback to the global model.
   Inspect `per_municipio_metrics.parquet` to identify candidates.

2. **Inference / scoring API.** The current `train` entrypoint loads a single Pipeline. After
   refactor, scoring code must load the models dict and dispatch by `codigo_municipio`. Document
   this contract in the scoring module before merging.

3. **MLflow run structure.** Two options:
   - **Flat run** (simpler): one MLflow run, 35 × 3 = 105 per-municipio metrics as named params.
   - **Nested runs** (cleaner UI): parent run for the experiment; one child run per municipio.
     Requires `mlflow.start_run(nested=True)` inside the training loop. Adds complexity but
     makes per-municipio comparison easy in the MLflow UI.

4. **Lookback window.** 90 days captures ~1.3 annual cycles of the weekly frequency but only
   ~25 % of the annual cycle for the `sin(2π·doy/365)` channel. Channel 5-6 in the sequence
   already encode annual position, so 90 days may be sufficient. However, extending to 182 or 365
   days would give MiniRocket access to the raw historical trajectory at annual lags, which could
   help for municipios with strong seasonality. This requires refeaturization.

5. **Target transform.** Accident counts follow a Poisson-like distribution (non-negative,
   right-skewed, variance ≈ mean). The current z-score normalisation reduces bias but does not
   address skew. A `log1p` transform applied before z-scoring (and `expm1` in the inverse) would
   pull in extreme counts and may reduce RMSE on high-accident days. Low-risk to try alongside
   the per-municipio refactor.

---

## 9. Recommended Implementation Order

1. **Immediately** (no refactor needed): bump `svd_components: 1000` in `experiment.yaml` and
   re-run `make municipio-day-train`. Measures how much of the R² gap is due to SVD truncation.

2. **Refactor to Option A** (per-municipio Ridge, shared MiniRocket):
   - Add `_assemble_per_municipio_arrays` helper
   - Rewrite `featurize()` to store per-municipio sequence arrays
   - Rewrite `train()` loop (remove `Pipeline`, `TruncatedSVD`)
   - Update `evaluate()`, `persist()`, `mlflow_params()`
   - Remove `svd_components` from config

3. **After benchmarking**: if per-municipio R² variance is high, investigate the `log1p` target
   transform and/or extending `lookback_days` to 182.

4. **Deferred**: hierarchical / partial-pooling models (Option C), weather feature channels,
   per-municipio MLflow nested runs.
