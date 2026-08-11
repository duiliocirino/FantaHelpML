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
  - Stage 0: `data/raw/{SEASON}/Quotazioni_*.xlsx`, `data/utils/player_dob.csv`, `data/historical/*` → `data/intermediate/{SEASON}/data_preprocess_merge.xlsx`
  - Stage 1: reads above → `data/intermediate/{SEASON}/output_rp.csv`
  - Stage 2A: reads `output_rp.csv` + all `data/raw/{SEASON}/Rose_*.xlsx` → `models/fvm_distribution/{SEASON}/`
  - Stage 2B: reads `output_rp.csv` + models → `data/final/{SEASON}/players.csv`
* Add validation block at end of each notebook printing a small report and writing `data/intermediate/{SEASON}/validation_{stage}.json`.

## Phase 1 – Stage 0 data_preprocess_merge
**Focus:** clean ingestion and historical merge.

Improvements:
* Document mapping dict FantaHelp → Fantacalcio.
* Age computed from DOB, no manual increment.
* Historical FVM merged per season from data/raw/historical/* for seasons 22-23 onwards, columns FVM{season}
* Validation checks:
  - Row count matches Quotazioni
  - Id unique
  - Age computed for >99% players via DOB
  - Historical stats merged for players present in previous season
  - Historical FVM columns present for seasons 22-23+

Manual intervention remains after Stage 0.

## Phase 2 – Stage 1 ratings
**Focus:** stability of ExpectedMf and MyRating.

Improvements:
* Parameterise thresholds: min_matches, per-role matches_filter, FVM curve minimum games, age prediction branch >=16 games. ✅
* Hybrid blend: `ExpectedMf = w_fvm × FVM_pred + (1 - w_fvm) × hist_perf` ✅
  - `w_fvm = 0.75` (configurable)
  - `hist_perf = Mf_last + age_modifier_weight × age_delta` if player has recent data
  - Fallback: age curve prediction → FVM prediction
* FVM modelling upgrade using historical FVM ✅
  - Use per-season FVM{season} from Stage 0, introduced from 22-23.
  - Pooled same-season weighted linear regression per Role_M: FVM_t → Mf_t on all seasons ≥22-23
  - Observation weight: `clip(Pg, 0, 38) × exp(-season_decay × season_index)`
  - Merge Ds into Dd for modelling to increase sample size
  - `season_decay = 0.3` (exponential recency decay)
* Age curve: quadratic per role, used as delta modifier (`age_modifier_weight = 0.2`) and fallback prior ✅
* Pending improvements:
  - Option B: ΔFVM = FVM_t - FVM_{t-1} as additional feature
  - Option C: mixed-effects model with player random intercept
  - Non-linear FVM (degree 2, monotonic spline)
  - Role-specific blend weights
* Validation:
  - ExpectedMf range per role ✅
  - JSON report to `validation_stage1.json` ✅
  - FVM model hold-out CV per role, MAE/RMSE vs single-season baseline (pending)

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
* 25-26 trial: use Quotazioni 25-26 + player_dob.csv + historical stats up to 24-25. Train Stage 2A on 24-25 auction data, predict 25-26 prices. Validate against 25-26 auction data when available.
* 26-27 production: repeat with 25-26 auction data training for 26-27.
* Optional public auction data from Italy can be used for in-season calibration, not for training production models until validated.

## Validation / Test plan per stage
Stage 0: data integrity report
Stage 1: rating sanity report
Stage 2A: model metrics report + hold-out league performance
Stage 2B: BE contract compliance report

All reports saved under `data/intermediate/{SEASON}/validation_*.json` for reproducibility.

## Progress (25-26 trial)
* Stage 0: SEASON helpers, repo-root detection, DOB-based Age, missing DOB export, historical FVM merge, validation JSON ✅
* Stage 1: SEASON helpers, repo-root detection, configurable parameters, hybrid ExpectedMf blend, weighted FVM regression, season recency decay, Ds→Dd merge, role-colored visualizations, explanatory markdown, per-role distribution validation ✅
* Stage 2A: pending
* Stage 2B: pending

## Next immediate actions
1. Stage 1 — Phase C1: test non-linear FVM (degree 2, monotonic spline) vs linear baseline
2. Stage 1 — Phase C3: test ΔFVM feature for players with historical FVM
3. Stage 1 — Phase E: role-specific blend weights grid search
4. Apply same clean structure to Stage 2A and Stage 2B.
5. Execute 25-26 trial Stage 0 → 1 → 2A → 2B with validation reports.
6. Document findings and decide on model upgrade for 26-27.
