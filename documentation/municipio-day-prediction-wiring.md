# Municipio-Day Prediction Wiring

This note systematizes what is still missing to make municipio-day predictions
available in the visualization app and to let the full ML flow consume both the
batch ETL output and a cached near-real-time (NRT) overlay.

## Current State

- Historical visualization data exists under
  `data/gold/br101_sc_municipio_panel`.
- The Streamlit app in `src/viz` reads the daily and weekly gold panels and can
  append future rows from `VIZ_FORECAST_PATH` or from one of:
  - `data/gold/ml/municipio_day_forecast`
  - `data/gold/ml/municipio_day_predictions`
- The app expects forecast rows to have at least:
  - `codigo_municipio`
  - a date column: `accident_date`, `date`, `data`, `prediction_date`, or `ds`
  - a value column: `accident_count`, `predicted_accident_count`, `prediction`,
    `yhat`, or `forecast`
- The municipio-day ML module has only two CLI phases:
  - `featurize`
  - `train`
- Trained municipio-day artifacts currently exist under
  `data/gold/ml/municipio_day_regression` and the weather variant under
  `data/gold/ml/municipio_day_regression_weather`.
- The existing Open-Meteo job writes/upserts
  `gold/br101_sc_municipio_panel/daily_weather` by
  `(codigo_municipio, weather_date)`.
- `Makefile` has targets for featurization and training, but no prediction
  target.
- `dvc.yaml` currently wires the batch pipeline through
  `municipio_day_featurize`; it does not wire municipio-day training or
  prediction.

## Main Gaps

1. There is no prediction phase in `src/ml/municipio_day_regression.py`.
2. There is no materialized forecast table for the viz app to consume.
3. Future feature generation is not implemented.
4. Recursive multi-day forecasting is not implemented.
5. The NRT PRF accident cache/overlay is not defined.
6. Weather refresh currently has a Make target that stops at today, which is not
   enough for a weather-dependent forecast horizon.
7. The viz app can append daily predictions, but weekly prediction display needs
   an explicit aggregation/contract.
8. There is no manifest tying a forecast output to model artifacts, source data
   cutoff, NRT cache watermark, or generation timestamp.

## Forecast Data Contract

Use a daily forecast table as the source of truth:

`gold/ml/municipio_day_forecast`

Required columns:

- `codigo_municipio`: string IBGE municipio code.
- `nome_municipio`: string municipio name.
- `accident_date`: date being predicted.
- `predicted_accident_count`: non-negative numeric prediction for display.
- `prediction_raw`: raw model output before clipping or post-processing.
- `forecast_horizon_day`: integer offset from the historical/NRT cutoff.
- `model_variant`: for example `with_weather` or `no_weather`.
- `model_artifact_subpath`: source artifact directory used for prediction.
- `source_panel_max_date`: latest accident date used as observed input.
- `generated_at`: timestamp for the prediction run.

Recommended optional columns:

- `prediction_lower`
- `prediction_upper`
- `run_id`
- `feature_input_subpath`
- `weather_input_subpath`
- `nrt_cache_watermark`
- `fonte`

The viz app already treats future rows as `fonte = "previsao"` internally. The
producer should still write enough metadata for debugging and audits.

Weekly display should be derived from the daily forecast table unless a separate
weekly model is introduced. The app should aggregate future daily predictions by
`codigo_municipio` and Monday `week_start`, summing `predicted_accident_count`
into the displayed `accident_count`.

## Prediction Phase Needed

Add a third CLI phase:

```bash
python -m src.ml.municipio_day_regression predict
```

Recommended config/env additions:

- `ML_MUNICIPIO_DAY_MODEL_INPUT_SUBPATH`
- `ML_MUNICIPIO_DAY_FEATURE_INPUT_SUBPATH`
- `ML_MUNICIPIO_DAY_FORECAST_OUTPUT_SUBPATH`
- `ML_MUNICIPIO_DAY_FORECAST_HORIZON_DAYS`
- `ML_MUNICIPIO_DAY_INCLUDE_WEATHER_FEATURES`

The prediction phase should:

1. Load `panel_with_features.parquet` and `feature_registry.json`.
2. Load `minirocket_transformer.joblib` and `ridge_model.joblib`.
3. Resolve the observed cutoff date from the combined batch plus NRT feature
   panel.
4. Generate future rows for every in-scope municipio and every future date in
   the requested horizon.
5. Build the same sequence channels used by training.
6. Predict one day at a time per municipio.
7. For horizons greater than one day, append each prior prediction back into the
   accident-count channel so the next lookback window is coherent.
8. Clip display predictions to `>= 0`, while preserving `prediction_raw`.
9. Persist `gold/ml/municipio_day_forecast`.
10. Persist a small `forecast_manifest.json`.

The model predicts scaled values per municipio. The prediction code must reuse
the same training-period mean/std logic currently embedded in sequence
construction so inverse scaling matches training and evaluation.

## Future Feature Generation

Future rows need the same exogenous columns as the training panel:

- `codigo_municipio`
- `nome_municipio`
- `sg_uf`
- `cd_rgint`
- `nm_rgint`
- `accident_date`
- `year`
- `month`
- `day`
- `accident_count`
- `split`
- `non_working_day_weight`
- `days_off_ahead`
- `days_off_before`
- weather columns when using the weather model

For future dates:

- Calendar and holiday features can be computed deterministically with the same
  helper functions already present in `municipio_day_regression.py`.
- `split` should use a non-training label such as `forecast`.
- `accident_count` starts unknown and is filled recursively with prior
  predictions.
- Weather features must come from `daily_weather` for every
  `(codigo_municipio, accident_date)` in the forecast horizon when using the
  weather model.
- The no-weather model should remain available as a fallback when future weather
  coverage is incomplete.

## Batch Plus NRT Data Layer

The prediction input should be a combined observed panel:

`observed_panel = batch_gold_panel upserted with nrt_gold_overlay`

Missing pieces:

1. Define the PRF NRT source and a local cache path, for example:
   `data/cache/nrt/prf_accidents`.
2. Normalize the NRT accident records into the same schema as
   `silver/prf_accidents_standardized`.
3. Create a gold NRT overlay table, for example:
   `gold/br101_sc_municipio_panel_nrt/canonical_accidents_by_municipio_day`.
4. Upsert by a stable accident key. If the source does not expose one, define a
   deterministic hash over source fields and document collision risk.
5. Produce a combined feature panel under a separate path, for example:
   `gold/ml/municipio_day_features_nrt`.
6. Track watermarks:
   - latest batch accident date
   - latest NRT accident timestamp/date
   - NRT fetch timestamp
   - latest weather date

Do not mutate the DVC-controlled historical gold panel for ad hoc NRT refreshes.
Keep NRT outputs separate or explicitly materialized as a cache/overlay.

## Weather Horizon

For weather-dependent forecasts, `daily_weather` must include future dates for
the requested horizon.

Current Open-Meteo behavior:

- `--start-date-from-existing-max` can continue from the current max weather
  date.
- If no end date is supplied, the job defaults to `start_date + 15 days`.
- The current `openmeteo-api2gold` Make target uses `--end-date-today`, so it
  does not prepare a future weather horizon.

Needed changes:

- Add a Make target for forecast weather refresh, for example
  `openmeteo-api2gold-forecast`.
- Either rely on the existing default horizon or add an explicit
  `--forecast-horizon-days` CLI argument.
- Validate full coverage before running the weather model:
  `n_municipios * horizon_days` rows must exist for future dates.
- If coverage fails, either abort or route to the no-weather model.

## Pipeline Wiring

Recommended Make targets:

- `municipio-day-predict`
- `municipio-day-predict-weather`
- `municipio-day-predict-no-weather`
- `municipio-day-refresh-nrt`
- `municipio-day-refresh-weather-forecast`

Recommended local sequence:

```bash
make br101-sc-municipio-silver2gold
make openmeteo-api2gold-forecast
make municipio-day-featurize-weather
make municipio-day-train-weather
make municipio-day-predict-weather
docker compose up viz
```

For NRT prediction:

```bash
make municipio-day-refresh-nrt
make municipio-day-refresh-weather-forecast
make municipio-day-featurize-nrt-weather
make municipio-day-predict-weather
docker compose up viz
```

DVC decision:

- Keep fully reproducible batch ETL and batch featurization in DVC.
- Add training to DVC only if model artifacts are intended to be reproducible
  repo artifacts.
- Keep NRT forecast generation outside DVC unless the NRT cache is materialized
  as an explicit DVC dependency.
- If NRT is added to DVC, use separate stages/outs so the historical gold panel
  remains stable.

## Visualization Changes Still Needed

1. Set `VIZ_FORECAST_PATH=/app/data/gold/ml/municipio_day_forecast` explicitly in
   `docker-compose.yml` once the producer path is finalized.
2. Aggregate daily predictions for weekly display instead of expecting weekly
   prediction rows.
3. Show whether the selected period is historical, forecast, or mixed.
4. Display forecast metadata in the sidebar:
   - generated at
   - model variant
   - source cutoff date
   - forecast horizon
5. Optionally expose uncertainty columns when available.
6. Keep the current empty-forecast behavior, but make the absence of future
   predictions visible in the UI when a forecast path is configured and empty.

## Validation

Add tests for:

- Forecast schema normalization in `src/viz/data.py`.
- Weekly aggregation of daily forecast rows.
- Future calendar/holiday feature generation.
- Weather coverage validation.
- Recursive forecast loop with a small synthetic panel and fake model.
- Non-negative display predictions while retaining raw predictions.

Add integration smoke checks:

1. Run featurization.
2. Run training or use existing artifacts.
3. Run prediction with a small horizon.
4. Assert `gold/ml/municipio_day_forecast` exists.
5. Assert it contains one row per in-scope municipio per future day.
6. Assert the max forecast date is greater than the max historical date.
7. Start `viz` and confirm future dates appear in the period slider.

## Suggested Implementation Order

1. Finalize and document the forecast table schema.
2. Update viz data loading to aggregate daily forecasts for weekly mode.
3. Implement `predict` for the no-weather model first.
4. Add forecast manifests and Make targets.
5. Extend weather refresh for future horizon coverage.
6. Implement weather-model prediction with fallback to no-weather.
7. Define and build the PRF NRT cache and overlay.
8. Add NRT featurization output paths.
9. Decide whether training/prediction belong in DVC or only in Make/orchestration.
10. Add smoke tests for end-to-end prediction visibility in Streamlit.

