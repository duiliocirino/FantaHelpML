# Stage 01 Ratings Improvement Plan

## Objective
Make `ExpectedMf` less conservative and more predictive for 25-26 trial and 26-27 production while keeping BE contract and validation gates.

## Current baseline
* FVM → Mf pooled same-season weighted linear regression per Role_M, Ds merged into Dd
* Observation weight: `clip(Pg, 0, 38) × exp(-season_decay × season_index)`
* Age curve: quadratic mean Mf per age per role, used as delta modifier and fallback prior
* Hybrid blend: `ExpectedMf = w_fvm × FVM_pred + (1 - w_fvm) × hist_perf`
  - `w_fvm = 0.75`
  - `hist_perf = Mf_last + age_modifier_weight × age_delta` (if player has recent data)
  - Fallback: age curve prediction → FVM prediction (if no recent data / no age curve)
  - `age_modifier_weight = 0.2`
  - `season_decay = 0.3` (exponential recency decay on both age curve and FVM regression)
* Filters: `fvm_curve_min_games = 22`, `age_pred_min_games = 16`, per-role `matches_filter`
* Validation: ExpectedMf range per role, JSON report to `validation_stage1.json`

## Principles
* One change at a time, measure impact
* Role-specific tuning allowed, global defaults kept
* All experiments logged in `data/intermediate/25-26/validation_stage1_*.json`
* No commits from agent

## Phase A – Baseline measurement
A1. Run Stage 0 → Stage 1 end-to-end for 25-26 with current clean notebook
A2. Save baseline metrics:
  - Per-role FVM-Mf correlation, RMSE, MAE on hold-out season 24-25
  - ExpectedMf distribution per role: mean, std, 5th/95th percentile
  - Top 20 players spot-check vs manual expectation
A3. Export `validation_stage1_baseline.json`

## Phase B – Age usage simplification
B1. Reduce age role to modifier only
  - Keep age curve for delta only, remove age-only prior
  - New blend: `ExpectedMf = FVM_pred + w_age * age_delta` where `age_delta = expected_diff_performance(Age)`
  - Test `w_age ∈ {0.0, 0.2, 0.4}`
B2. Replace polynomial age curve with empirical age-to-age deltas per role
  - Compute `Δ_age = mean Mf_age - mean Mf_age-1` from pooled data
  - Cap delta to [-3, +3] points
B3. Measure impact on ExpectedMf spread per role

Acceptance: Std of ExpectedMf per role increases vs baseline without increasing MAE on hold-out

## Phase C – FVM model improvements
C1. Non-linear FVM
  - Current: linear poly1d
  - Test: Polynomial degree 2, monotonic UnivariateSpline s=...
  - Keep same-season pooling
C2. Weight by games
  - Fit FVM model with `sample_weight = Pg` capped at 38
  - Compare linear weighted vs unweighted
C3. ΔFVM feature
  - Compute `ΔFVM = FVM_t - FVM_{t-1}` for players with historical FVM
  - Model options:
    1. `Mf ~ FVM`
    2. `Mf ~ FVM + ΔFVM`
    3. `Mf ~ FVM * (1 + k*ΔFVM)`
C4. Per-season vs pooled
  - Pooled baseline
  - Rolling 2-season window model per role
  - Measure stability of coefficients

Acceptance: MAE on hold-out 24-25 decreases for ≥ 7/11 roles

## Phase D – Mixed effects & player persistence
D1. Player random intercept prototype
  - Fit `Mf ~ FVM + Age + (1|Player)` per role using statsmodels MixedLM
  - Compare predictions vs pooled
D2. Player-specific shrinkage
  - For players with ≥2 seasons, use `player_mean_Mf` shrunk toward role mean
  - Blend with FVM prediction

Acceptance: Reduction in within-player prediction error

## Phase E – Blend tuning
E1. Role-specific blend weights
  - High FVM correlation roles: A, T, Pc → higher FVM weight
  - Low correlation roles: Dd, Ds → lower FVM weight
E2. Grid search
  - `rating_blend_fvm ∈ {0.3,0.5,0.7}` per role
  - Keep age delta weight small

## Phase F – Validation and documentation
F1. Final validation report
  - Per-role ExpectedMf range, correlation, RMSE
  - Hold-out MAE comparison baseline vs best config
  - Spot-check top 20 players
F2. Update `docs/improvement-roadmap.md` with chosen config
F3. Parameterise all knobs in notebook first cell:
  - `fvm_model_degree`
  - `fvm_weight_games`
  - `use_delta_fvm`
  - `age_modifier_weight`
  - `rating_blend_fvm_per_role`

## Experiment tracking
Each experiment creates:
`data/intermediate/25-26/validation_stage1_exp_<name>.json`
with keys: `experiment_name`, `params`, `per_role_mae`, `per_role_rmse`, `expectedmf_std`, `top20_diff`

## Status
| Phase | Item | Status |
|---|---|---|
| A | Baseline measurement | ✅ Done — 532 rows, correlations documented |
| B | Age modifier reduction (`age_modifier_weight = 0.2`) | ✅ Done |
| B | Season recency decay (`season_decay = 0.3`) | ✅ Done |
| C | Ds → Dd merge | ✅ Done |
| C | Historical FVM pooling (all seasons ≥22-23) | ✅ Done |
| C | C2 — Weight by games (`clip(Pg, 0, 38)`) | ✅ Done |
| C | C1 — Non-linear FVM (degree 2, spline) | ⏳ Pending |
| C | C3 — ΔFVM feature | ⏳ Pending |
| C | C4 — Per-season vs pooled comparison | ⏳ Pending |
| D | Mixed-effects prototype | ⏳ Pending (insufficient within-player replication) |
| E | Role-specific blend weights | ⏳ Pending |
| F | Final validation + parameter documentation | ⏳ Pending |

## Next immediate action
1. Phase C1 — Test non-linear FVM (degree 2, monotonic spline) vs linear baseline
2. Phase C3 — Test ΔFVM feature for players with historical FVM
3. Phase E — Role-specific blend weights grid search
4. Phase F — Final validation report

## Risks
* Overfitting with degree 2 FVM on small roles
* ΔFVM missing for new players → fallback to FVM only
* Mixed-effects requires more compute and careful convergence
