# FantaHelpML Improvement Roadmap

## Context
Current pipeline Stage 0 → 1 → 2A → 2B is functional for 24-25. Goal is to clean, standardise and improve the pipeline for 25-26 trial and 26-27 production.

Key constraints:
* No commits from agent.
* Manual step after Stage 0: Age / Regularness / Mate update.
* BE contract columns must be preserved.
* Names are handled correctly, no normalisation needed.

## Guiding principles
* Standardise inputs/outputs per notebook, SEASON variable everywhere.
* Validate each stage with explicit checks and metrics.
* Improve price modelling by using multi-league auction data and exploring non-linear FVM → price relationship.
* Keep FVM as primary proxy but test alternatives with real auction data.

## Phase 0 – Standardisation & validation scaffolding
**Objective:** Make notebooks deterministic and testable.

* Add first cell: `SEASON = "xx-yy"` and path helpers RAW_DIR, INTER_DIR, FINAL_DIR, MODELS_DIR.
* Standardise IO:
  - Stage 0: `data/raw/{SEASON}/Quotazioni_*.xlsx`, `copyCsvReal.xlsx`, `data/historical/*` → `data/intermediate/{SEASON}/data_preprocess_merge.xlsx`
  - Stage 1: reads above → `data/intermediate/{SEASON}/output_rp.csv`
  - Stage 2A: reads `output_rp.csv` + all `data/raw/{SEASON}/Rose_*.xlsx` → `models/fvm_distribution/{SEASON}/`
  - Stage 2B: reads `output_rp.csv` + models → `data/final/{SEASON}/players.csv`
* Add validation block at end of each notebook printing a small report and writing `data/intermediate/{SEASON}/validation_{stage}.json`.

## Phase 1 – Stage 0 data_preprocess_merge
**Focus:** clean ingestion and historical merge.

Improvements:
* Document mapping dict FantaHelp → Fantacalcio.
* Parameterise age increment logic, explicit missing handling.
* Validation checks:
  - Row count matches Quotazioni
  - Id unique
  - Age incremented vs previous season for returning players
  - Historical stats merged for players present in previous season

Manual intervention remains after Stage 0.

## Phase 2 – Stage 1 ratings
**Focus:** stability of ExpectedMf and MyRating.

Improvements:
* Parameterise thresholds: min_matches, per-role matches_filter, FVM curve minimum games, age prediction branch >=16 games.
* Make curve fitting reproducible with random_state where applicable.
* Validation:
  - ExpectedMf range per role
  - MyRating in [1,5]
  - No NaNs in FVM/Price for active players
  - Distribution comparison vs previous season

## Phase 3 – Stage 2A fvm_to_distribution – multi-league price modelling
**Focus:** better auction price mean/std estimation.

Current: single league LinearRegression FVM → price, count>=8.

Improvements:
* Aggregate multiple Rose files in `data/raw/{SEASON}/`. Use previous season auction data to train for next season, as current practice.
* Keep FVM as primary feature, explore:
  - Non-linear models: PolynomialFeatures + Ridge, Gradient Boosting, Quantile Regression for std
  - Log transform price
  - Role-specific models
* Validation:
  - Training data coverage per role and FVM bin
  - Hold-out league CV: MAE/RMSE for mean, std calibration
  - Residual analysis vs FVM
* Exploration task: test how well 25-26 FVM → price model trained on 24-25 auction data predicts 25-26 auction prices. Measure drift.

## Phase 4 – Stage 2B regressors – final export
**Focus:** BE-ready quality.

* Keep FVM-only baseline for production, compare multi-feature models offline.
* Ensure final CSV schema matches BE contract.
* Validation:
  - Schema check
  - expprice/expstd >0
  - No duplicate Id
  - Spot-check top players

## Data strategy
* 25-26 trial: use Quotazioni 25-26 + copyCsvReal 25-26 + historical stats. Train Stage 2A on 24-25 auction data, predict 25-26 prices. Validate against 25-26 auction data when available.
* 26-27 production: repeat with 25-26 auction data training for 26-27.
* Optional public auction data from Italy can be used for in-season calibration, not for training production models until validated.

## Validation / Test plan per stage
Stage 0: data integrity report
Stage 1: rating sanity report
Stage 2A: model metrics report + hold-out league performance
Stage 2B: BE contract compliance report

All reports saved under `data/intermediate/{SEASON}/validation_*.json` for reproducibility.

## Next immediate actions
1. Gather 25-26 raw files and place in `data/raw/25-26/`.
2. Run Phase 0 standardisation on notebooks.
3. Execute 25-26 trial Stage 0 → 1 → 2A → 2B with validation reports.
4. Document findings and decide on model upgrade for 26-27.
