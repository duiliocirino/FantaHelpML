# Stage 01 Ratings Improvement Plan

## Objective
Make `ExpectedMf` less conservative and more predictive for 25-26 trial and 26-27 production while keeping BE contract and validation gates.

## Current baseline
* FVM → Mf pooled same-season weighted regression per Role_M, Ds merged into Dd
* **Transform**: per-role — `sqrt(FVM)` for P/D, `linear(FVM)` for C/A. Compresses extreme defender FVM, preserves attacker spread. Fitted via weighted OLS.
* **fvm_scale**: per-role multiplier on FVM prediction to counteract conservative regression: `{'P': 1.0, 'D': 1.05, 'C': 1.05, 'A': 1.05}`
* Observation weight: `clip(Pg, 0, 38)`
* Age curve: quadratic mean Mf per age per role, used as delta modifier and fallback prior
* Hybrid blend: `ExpectedMf = w_fvm × FVM_pred + (1 - w_fvm) × hist_perf`
  - `w_fvm` is **role-specific**: `{'P': 0.8, 'D': 0.7, 'C': 0.8, 'A': 0.8}`
  - `hist_perf = Mf_last + age_modifier_weight × age_delta` (if player has recent data)
  - Fallback: age curve prediction → FVM prediction (if no recent data / no age curve)
  - `age_modifier_weight = 0.4`
  - `season_decay = 0.3` (exponential recency decay on age curve)
* Filters: `age_pred_min_games = 16`, per-role `matches_filter`
* Validation: ExpectedMf range per role, JSON report to `fvm_tuning_report.json`

## Calibration note (IMPORTANT)
The `fvm_scale` values and blend weights were fine-tuned against **season 25-26 actual performances** as ground truth targets:
- Attackers: Martinez L. ~8.15, Malen ~8.15, Hojlund/Yildiz ~7.6, Raspadori/Nkunku ~7.0
- Midfielders: Orsolini ~7.7, Pulisic/Calhanoglu ~7.3, Zaniolo/Baturina ~7.0
- Defenders: Dimarco ~7.8, Palestra/Dumfries ~6.8, Bremer ~6.5

**When moving to season 26-27**, these targets no longer apply. The config structure (sqrt/linear transforms, role-specific weights) should remain valid, but `fvm_scale` values may need re-calibration once new ground truth is available. Always re-validate against actual results from the most recent completed season.

### Known limitation: extreme FVM outliers
Players with FVM values far above the pack (e.g., 300+) may still see inflated ExpectedMf despite sqrt transform and role-specific scaling. The sqrt compresses the range but doesn't eliminate the leverage of extreme points on the fitted slope. Mitigations available via config:
- `fvm_cap` in candidate config — caps FVM during training to prevent slope inflation
- Lower `w_fvm` for the affected major role — lets historical performance dominate the blend
- Increase `fvm_scale` downward (e.g., 0.95) for that role — directly scales down FVM predictions

These are trade-offs, not bugs — capping or down-weighting may compress spread for mid-range players too. Monitor per-role ExpectedMf std after any adjustment.

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

## Phase G — Tuning framework (NEW)
G1. Build `pipeline/fvm_tuning.py` — automated hold-out validation framework
  - Trains multiple FVM configs on all-but-last FVM season
  - Validates on most recent FVM season (hold-out)
  - Selects best config by lowest MAE
  - Retrains on all data, saves report
G2. Integrate into notebook as preliminary step (cells 4-5)
  - `OVERRIDE_CONFIG` variable for manual override
  - Report saved to `data/intermediate/{SEASON}/fvm_tuning_report.json`
G3. Candidate configs tested (10 total):
  - `linear_baseline`, `linear_w0.9`, `linear_w1.0`
  - `linear_cap200`, `linear_cap200_w0.9` (FVM capped during training)
  - `theilsen`, `theilsen_cap200` (robust regression)
  - `sqrt_baseline`, `sqrt_role_weights` (sqrt FVM transform + role-specific blend weights)
  - `sqrt_pd_linear_ac` (per-role sqrt/linear + fvm_scale + role-specific weights)

Acceptance: Framework runs end-to-end, produces report, notebook uses selected config

## Status
| Phase | Item | Status |
|---|---|---|
| A | Baseline measurement | ✅ Done — 532 rows, correlations documented |
| B | Age modifier reduction (`age_modifier_weight = 0.4`) | ✅ Done |
| B | Season recency decay (`season_decay = 0.3`) | ✅ Done |
| C | Ds → Dd merge | ✅ Done |
| C | Historical FVM pooling (all seasons ≥22-23) | ✅ Done |
| C | C2 — Weight by games (`clip(Pg, 0, 38)`) | ✅ Done |
| C | C1 — Non-linear FVM (degree 2, spline) | ❌ Rejected — overfits, worse RMSE |
| C | C1 — Per-role sqrt/linear transform | ✅ Done — `sqrt_pd_linear_ac` active override |
| C | C3 — ΔFVM feature | ⏳ Pending |
| C | C4 — Per-season vs pooled comparison | ⏳ Pending |
| D | Mixed-effects prototype | ❌ Rejected — insufficient within-player replication |
| E | E1 — Role-specific blend weights (`{'P':0.8,'D':0.7,'C':0.8,'A':0.8}`) | ✅ Done |
| E | fvm_scale — per-role FVM multiplier (`{'P':1.0,'D':1.05,'C':1.05,'A':1.05}`) | ✅ Done — calibrated on 25-26 actuals |
| F | Final validation + parameter documentation | ⏳ Pending |
| G | G1 — Tuning framework (`fvm_tuning.py`) | ✅ Done |
| G | G2 — Notebook integration | ✅ Done |
| G | G3 — Hold-out validation (10 configs) | ✅ Done — `sqrt_pd_linear_ac` active override |
| G | Config centralization (`pipeline/config/stage01.yaml`) | ✅ Done |

## Hold-out validation findings (24-25)
| Config | MAE | RMSE | Notes |
|---|---|---|---|
| linear_baseline | 0.269 | 0.362 | Lowest MAE but conservative spread |
| linear_cap200 | 0.305 | 0.452 | Over-predicts Pc (MAE 0.691) and A |
| theilsen | 0.300 | 0.403 | More robust but higher MAE overall |
| sqrt_baseline | 0.275 | — | Slight MAE increase, better defender spread |
| sqrt_pd_linear_ac | 0.276 | 0.370 | **Active override** — per-role transforms + fvm_scale |

**Key insight**: The hold-out validation objectively selects the baseline linear model by raw MAE.
However, `sqrt_pd_linear_ac` provides better defender spread (sqrt compresses extreme FVM,
linear preserves attacker spread) at a small MAE cost (0.269→0.276). Role-specific blend weights
let history breathe more for defenders (w_fvm=0.7) while keeping midfielders/attackers tight to FVM
(w_fvm=0.8). The `fvm_scale` multiplier (1.05 for D/C/A) counteracts conservative regression.
Active via `override_config: sqrt_pd_linear_ac` in YAML.

## Next immediate action
1. Phase C3 — Test ΔFVM feature for players with historical FVM
2. Phase C4 — Per-season vs pooled comparison
3. Phase F — Final validation report
4. Extend tuning framework with ΔFVM configs

## Risks
* ΔFVM missing for new players → fallback to FVM only
* Hold-out may not generalize to future seasons (only 2 training seasons available)
* Capped FVM models over-predict for elite players when applied to uncapped FVM
