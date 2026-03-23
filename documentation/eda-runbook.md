# BR-101 EDA Runbook

This document consolidates the intent under the notebook directory. This runbook describes a recommended order of operations.

## 1. Goal of the EDA

1. Define a "canonical BR-101 accident universe" from PRF occurrence-level tables and DNIT road geometry.
2. Produce spatial, temporal, and categorical summaries that support a later forecasting setup, likely at the `RGI x week` level.

That means the EDA should lock down:

- which accidents count as "on BR-101"
- which territorial unit each accident belongs to
- which years or state-year slices are excluded for missingness or coverage problems
- how severity is measured

## 2. Current Inputs and Artifacts

### Raw inputs currently present in `data/bronze`

- PRF occurrence tables: `2017` through `2026` as `*Agrupados por ocorrência.csv.gz`
- DNIT road geometry snapshots: `201703A.zip`, `202107A.zip`, `202601A.zip`
- IBGE municipalities: `BR_Municipios_2024.zip`, IBGE regiões imediatas: `BR_RG_Imediatas_2024.zip`

## 3. EDA Workflow

The current order is:

1. Standardize raw PRF tables.
2. Build the canonical BR-101 spatial corridor.
3. Build municipality and RGI attribution layers.
4. Assign accidents to the corridor and to territorial units.
5. Perform coverage checks and decide exclusions.
6. Run the substantive EDA on outcomes and covariates.
7. Only after those decisions, prepare the panel for modeling.

## 4. Step-by-Step Plan

### Step 0. Environment and reproducibility

Before running the EDA, make sure the local environment is consistent:

```bash
uv sync --all-groups
```

The analysis is staying notebook-based for now, following the order of operations:

1. road geometry and canonical corridor
2. territorial attribution layers
3. accident assignment and descriptive analysis

### Step 1. Standardize PRF occurrence and person tables

Minimum cleaning steps:

- concatenate yearly occurrence files
- normalize decimal commas in `km`, `latitude`, and `longitude`
- parse `data_inversa` and `horario` into a timestamp
- normalize `br` to canonical string form
- flag rows with missing or obviously invalid coordinates (set outlier coordinates to nan)

Suggested derived fields:

- `year`, `month`, `quarter`, `week_start`
- `is_br101_declared = (br == 101)`
- `has_valid_coords`
- `fatal_victims_occ = mortos` from occurrence table

### Step 2. Build the canonical BR-101 spatial corridor

The intent is as follows:

- take DNIT BR-101 geometry from multiple years
- harmonize CRS
- dissolve each year into a single road geometry
- buffer by 500 meters
- union across years to define the corridor within which accidents are considered to have happened on BR-101

Recommended implementation details:

- read only BR-101 features from each DNIT snapshot
- project to a meter-based CRS before buffering
- use the same projected CRS for all geometry operations
- dissolve within each year before buffering
- buffer by `500` meters
- union the three yearly buffered corridors
- persist both the yearly corridors and the final union

Tradeoffs:

- Union of all yearly traces captures historical realignments and may rescue accidents that fall on old alignments.
- That same union can also create an overly broad corridor where the road moved meaningfully over time.

Sensitivity choices worth considering:

- `latest-year only`: best if the study is about present-day geometry, but weaker for older accidents
- `union of all years`: best if the study is about any historical BR-101 alignment, but more permissive
- `core overlap or majority rule`: more conservative, but may incorrectly exclude valid accidents on changed segments

### Step 3. Build municipality and RGI attribution layers

The intended output is artifacts that can partition the BR-101 corridor into sections that map 1:1 to a municipality or RGI.

There are really two separate tasks here:

1. Select the territorial units through which BR-101 passes.
2. Create road sections that inherit exactly one territorial identifier.

Recommended process:

- load the IBGE municipality and RGI layer
- filter to the 12 UFs crossed by BR-101:
  - `PB`, `BA`, `PR`, `AL`, `PE`, `ES`, `RN`, `RS`, `SC`, `SE`, `SP`, `RJ`
- intersect the canonical road corridor with municipality polygons
- do the same for RGI polygons
- explode multipart intersections into atomic road sections
- compute road length per municipality and per RGI
- persist:
  - municipality polygons retained in scope
  - RGI polygons retained in scope
  - road-by-municipality sections
  - road-by-RGI sections

### Step 4. Define the canonical accident set

A practical classification is:

- `declared_and_inside`: `br == 101` and point intersects canonical corridor
- `declared_outside`: `br == 101` but point falls outside corridor
- `undeclared_inside`: `br != 101` but point falls inside corridor
- `outside_all`: neither declared nor spatially inside

Then choose one of these policies:

- strict: keep only `declared_and_inside`
- permissive: keep `declared_and_inside` plus reviewed `undeclared_inside`
- label-first: keep all `br == 101`, use corridor only for diagnostics

### Step 5. Attribute accidents to municipality and RGI

Once the canonical set is fixed:

- convert accident points to a GeoDataFrame
- spatially join to municipality sections and RGI sections
- inspect unassigned points
- quantify how many assignments depend on tolerant geometry rather than exact geometry

Outputs to persist:

- accident table with final municipality attribution
- accident table with final RGI attribution
- summary table of unmatched or ambiguous assignments

### Step 6. Coverage checks and exclusion rules

This is the first explicit EDA deliverable:

- total cases by UF
- fatal victims by UF
- total cases by year
- fatal victims by year

That should be done before modeling or significance testing.

Recommended diagnostics:

- accident counts by `UF x year`
- fatal victims by `UF x year`
- people involved by `UF x year`
- share of accidents with missing coordinates by `UF x year`
- share of declared BR-101 accidents falling outside the corridor by `UF x year`

### Step 7. EDA for the main categorical variables

- `dia_semana`
- `fase_dia`
- `condicao_metereologica`
- `holiday_status`

We intend to find answer for two outcomes:

- number of accidents
- percentage of fatal victims out of total people involved

Which comprise two statistical problems:

1. Accident frequency
2. Accident severity

Framing:

- accident frequency:
  - aggregate counts by time and territorial unit
  - use a count model such as Poisson or Negative Binomial, depending on overdispersion
- severity:
  - define fatality outcome based on `mortos` columns
  - model fatal victims or fatal-involved share using the `pessoas` column as the denominator

Reasonable first-pass models:

- counts: `accident_count ~ dia_semana + fase_dia + condicao_metereologica + holiday_status + year + UF`
- severity: `fatal_victim_share ~ dia_semana + fase_dia + condicao_metereologica + holiday_status + year + UF`

Potential complications:

- spatial clustering by municipality or RGI
- temporal autocorrelation
- rare categories with unstable estimates
- `holiday_status` varying by state, which means the feature should depend on the accident UF

### Step 8. Construct the forecasting panel only after EDA decisions are fixed

The current intuition is a DeepAR-like model on `RGI x week`.

That panel should be built only after the following are settled:

- canonical accident inclusion rule
- territorial assignment rule
- exclusion list for missingness or partial coverage
- severity definition

Recommended panel design:

- unit: `RGI x week_start`
- full grid from earliest to latest retained week
- fill missing combinations with zero accidents only after exclusions are finalized
- include lagged or known-in-advance exogenous features separately

Suggested targets:

- weekly accident count
- weekly fatal victim count

Candidate exogenous features:

- holiday indicators or holiday-status aggregates
- weather features aggregated at the week and RGI level
- possibly seasonality encodings

But this part is still blocked by upstream ambiguity, especially how climate data will be mapped to road sections or RGIs

## 5. Recommended Deliverables From the EDA

The EDA should produce the following durable outputs:

1. A canonical accident table for BR-101 with inclusion flags and assignment diagnostics.
2. Municipality and RGI road-section tables with lengths.
3. A coverage/exclusion report by `UF x year`.
4. Descriptive tables and plots for accident counts and fatal outcomes.
5. A modeling-ready `RGI x week` panel, if and only if the upstream decisions are finalized.

## 6. Main Shortcomings and Risks in the Current Approach

- The canonical accident definition is not yet fixed.
- The 500 meter corridor idea is reasonable but still arbitrary and should be sensitivity-tested.
- Unioning road traces across years can over-include old or shifted alignments.
- The current notebook implementation buffers in degrees, which is incorrect for the stated intent.
- Territorial expansion via Voronoi-clipped buffers may solve one attribution problem by introducing another.
- Manual exclusion of years or states can become post hoc cherry-picking unless the rule is documented.
- `holiday_status` has to be defined carefully for state holidays and possibly holiday eves/day-after logic.
- Severity analysis depends on the person table, which is not yet consistently integrated.
- Climate enrichment is still conceptual and may become a large side-project if station-to-road mapping is not constrained early.

## 7. Some Design Decisions

1. The canonical BR-101 universe is the intersection of `br == 101` and spatially inside the DNIT corridor.

2. The road corridor is the union of the 2017, 2021, and 2026 buffers

3. "Significant missing data" is going to be decided based on visual/manual review.

6. `2026` is going to be included in descriptive EDA even though it is a partial year.

7. For the severity outcome, our primary metric is fatal victims per people involved.   - 