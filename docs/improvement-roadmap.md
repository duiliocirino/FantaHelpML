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
  - Stage 2A: reads all `data/raw/{SEASON}/Rose_*.xlsx` (+ prior season) + per-season Quotazioni (FVM) → `models/fvm_distribution/{SEASON}/`
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
* All parameters centralized in `pipeline/config/stage01.yaml`. ✅
* Hybrid blend: `ExpectedMf = w_fvm × FVM_pred + (1 - w_fvm) × hist_perf` ✅
  - `w_fvm` is **role-specific**: `{'P': 0.8, 'D': 0.7, 'C': 0.8, 'A': 0.8}`
  - `hist_perf = Mf_last + age_modifier_weight × age_delta` if player has recent data
  - Fallback: age curve prediction → FVM prediction
* FVM modelling upgrade using historical FVM ✅
  - Use per-season FVM{season} from Stage 0, introduced from 22-23.
  - Pooled same-season weighted linear regression per Role_M: FVM_t → Mf_t on all seasons ≥22-23
  - Observation weight: `clip(Pg, 0, 38)`
  - **Per-role transform**: `sqrt(FVM)` for P/D, `linear(FVM)` for C/A
  - **fvm_scale**: per-role multiplier on FVM prediction (`{'P':1.0,'D':1.05,'C':1.05,'A':1.05}`)
  - Merge Ds into Dd for modelling to increase sample size
  - `season_decay = 0.3` (exponential recency decay on age curve)
  - `age_modifier_weight = 0.4`
* Automated hold-out tuning framework (`pipeline/fvm_tuning.py`) ✅
  - 10 candidate configs, validated on most recent FVM season
  - `OVERRIDE_CONFIG` in YAML for manual selection
  - Report saved to `fvm_tuning_report.json`
* Role mapping: E (esterno) → C, T (trequartista) → C, Pc (seconda punta) → A ✅
* Pending improvements:
  - ΔFVM = FVM_t - FVM_{t-1} as additional feature
  - Per-season vs pooled FVM comparison
* Validation:
  - ExpectedMf range per role ✅
  - JSON report to `validation_stage1.json` ✅
  - FVM model hold-out CV per role, MAE/RMSE ✅

## Phase 3 – Stage 2A fvm_to_distribution – multi-league price modelling
**Focus:** better auction price mean/std estimation. **Status: done for 25-26 trial** (see `docs/phase3_stage2a2b_plan.md` for the full spec and results).

* Multi-season training: 24-25 auction (weight 1.0) + 23-24 auction (weight 0.7) ✅
* Contemporaneous FVM pairing: each auction uses the FVM of its own season (Quotazioni) ✅
* Mean: `Ridge([sqrt(FVM), FVM])` per role (beats legacy baseline in 3/4 roles) ✅
* Std: binned empirical std + Ridge smoothing per role ✅
* Canonical multi-source auction loader (Rose xlsx + future `auction_*.csv`) ✅
* Validation: player hold-out, FVM-bin analysis, cross-season drift check (MAE 10.9), JSON report ✅

Open items (post-trial):
* Std-model MAE threshold (4.9–23.9 absolute vs legacy-scale "< 3" spec) — review
* P/A overfit-check sample sizes (hold-out n=9/19) — inconclusive, drift check corroborates
* Further model exploration (GBM, quantile regression, log price) — deferred, see Phase 4

## Phase 4 – Stage 2B regressors – final export
**Focus:** BE-ready quality. **Status: done for 25-26 trial.**

* FVM-only production path (`expprice`/`expstd` from Stage 2A models, floors, int rounding) ✅
* Residual correction experiment (ExpectedMf-based, 4 variants) — no variant beat baseline;
  production stays `residual_correction: none` ✅
* Final CSV: all 14 BE columns, schema/positive/duplicate checks, `validation_stage2b.json` ✅
* Legacy experiment cells moved to `explorations/legacy_regressor_experiments.ipynb` ✅
* Open: top-20 price plausibility — user visual check

## Phase 5 — External auction data study (website), 26-27 prep
**Objective:** Study player prices vs FVM from national online auction data across league formats, and assess it as an in-season calibration / training source for 26-27. Placement: first task right after the 25-26 trial.

* Deterministic HTML parser script: extract teams from auction pages → canonical auction CSV (`Role, Name, Team, Price`) + metadata (source, season, credits, players, format_tag, extraction date). Matches the Phase 3A league-format-tag schema so data drops into Stage 2A training once validated.
* Human validation gate (by design): league sanity check (some leagues are nonsensical) + player-name matching against the season roster (fuzzy-match report, unmatched names flagged for manual review)
* Analysis (in `explorations/`): price vs FVM per role and per league format (4/6/8/10 players × 500/800/1000 credits), drift vs local Rose data, league outlier detection
* Decision after analysis: use for in-season calibration or as training source for 26-27 (training only once validated)

## Data strategy
* Rose file convention: `data/raw/{SEASON}/Rose_*.xlsx` = most recent completed auction before {SEASON} starts (t-1 season auction). See `docs/phase3_stage2a2b_plan.md`.
* 25-26 trial: use Quotazioni 25-26 + player_dob.csv + historical stats up to 24-25. Train Stage 2A on the 24-25 season auction (in `data/raw/25-26/`, weight 1.0) + 23-24 season auction (in `data/raw/24-25/`, weight 0.7). Cross-season drift check per Phase 3C4.
* 26-27 production: repeat with 25-26 season auction data training for 26-27.
* External website auction data (Phase 5): in-season data for the current season — candidate for calibration, and for training 26-27 models once validated.

## Validation / Test plan per stage
Stage 0: data integrity report
Stage 1: rating sanity report
Stage 2A: model metrics report + hold-out league performance
Stage 2B: BE contract compliance report

All reports saved under `data/intermediate/{SEASON}/validation_*.json` for reproducibility.

## Decisions
* **MyRating — resolved: keep, auto-derive in Stage 1.** The FE displays the rating, so the column stays.
  History (verified in git): always auto-computed, never manual — old `ratings.ipynb` (Apr 2024) computed
  `MyRating = round(0.4 × fvm_model(log FVM) + 0.6 × stats_model(ExpectedMf), 1)` on a 1–5 scale with decimals.
  Dropped during the Stage 01 refactor (vestigial `myrating_ok: true` still hardcoded in the validation cell).
  Restoration spec (closed form, old absolute scale, per role): pooled historical Mf `mean_role`/`std_role`,
  `z = (ExpectedMf - mean_role) / std_role`, `MyRating = round(clip(1 + (z + 4) / 6.5 * 4, 1, 5), 1)`.
  See `docs/phase3_stage2a2b_plan.md` prerequisite P1.
* **mate / regularness — resolved: keep, manual for now.** The user still needs them; they were entered
  manually after Stage 0 (by-design manual intervention point). Automation is a deferred investigation
  (e.g. regularity from historical Pg/injury patterns, mate from squad data). Stage 0 must therefore
  create the columns again (empty, for manual fill) so the chain stays BE-contract-complete.

## Progress (25-26 trial)
* Stage 0: SEASON helpers, repo-root detection, DOB-based Age, missing DOB export, historical FVM merge, empty Mate/Regularness manual columns, validation JSON ✅
* Stage 1: SEASON helpers, repo-root detection, configurable parameters, hybrid ExpectedMf blend, weighted FVM regression, season recency decay, Ds→Dd merge, role-colored visualizations, explanatory markdown, per-role distribution validation, sqrt FVM transform, role-specific blend weights, hold-out tuning framework, MyRating restored (auto-derived) ✅
* Stage 2A: done — standardized notebook, canonical auction loader, contemporaneous per-season FVM pairing, multi-season recency weights, Ridge sqrt mean + binned std per role, hold-out/drift validation, JSON report ✅
* Stage 2B: done — standardized notebook, production inference, residual correction experiment (no win), BE-ready `data/final/25-26/players.csv` (532×14, all checks pass) ✅
* Trial end-to-end: Stage 0 → 1 → 2A → 2B executed with validation reports ✅

## Next immediate actions
1. User review: top-20 price plausibility, std-model MAE threshold, P/A overfit-check sample sizes
2. Manual Stage 0 step for 25-26: fill Mate/Regularness (then re-run Stage 1 → 2B)
3. Phase 5: external auction data study (website) for 26-27 — sample HTML provided at start
4. Stage 1 leftovers (C3 ΔFVM, C4 per-season vs pooled, F final report) — planned follow-up investigation
5. Document findings and decide on model upgrade for 26-27
