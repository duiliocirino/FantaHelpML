# Phase 3 — Stage 2A + 2B Detailed Action Plan

## Objective
Refactor `02_fvm_to_distribution.ipynb` (Stage 2A) and `03_regressors.ipynb` (Stage 2B) to produce better auction price predictions (expprice, expstd) while standardizing the pipeline for 25-26 trial and 26-27 production.

## Guiding Principles
- One change at a time, measure impact before committing
- Mathematical choices are justified and validated empirically
- All experiments leave traceable outputs (JSON reports, plots)
- Config centralized in `pipeline/config/stage2a2b.yaml`
- Notebook structure follows Stage 01 pattern: SEASON variable, repo-root detection, validation JSON

---

## Current State (As-Is)

### Stage 2A (`02_fvm_to_distribution.ipynb`)
- Hardcoded to `24-25`, no SEASON variable, no repo-root detection
- Reads a single Rose file: `Rose_fantalega-nicosia.xlsx`
- Groups by player name across teams → computes mean/std/count of auction prices
- Filters to `count >= 8`
- Merges FVM and Role from `output_rp.csv`
- Applies `FVM >= 10` floor only to D role (inconsistent)
- Trains 2 `LinearRegression` models per role (P, D, C, A):
  - `model_mean`: `LinearRegression([FVM] → mean_price)`
  - `model_std`: `LinearRegression([FVM] → std_price)`
- Saves 8 `.joblib` files to `models/fvm_distribution/24-25/`
- No visualizations, no validation, no config

### Stage 2B (`03_regressors.ipynb`)
- Cells 0-11: Production — loads 8 joblib models, predicts mean/std per player, saves BE-ready CSV with all 14 columns
- Cells 12-45: Experimental — GPR, Ridge, Lasso, SVM multi-feature experiments (not used in production)
- Hardcoded to `24-25`
- Imports `utility` from repo root
- Produces correct BE contract output

### Data Available
| File | Location | Content |
|---|---|---|
| Rose 24-25 | `data/raw/24-25/Rose_fantalega-nicosia.xlsx` | All transactions of the 23-24 season auction (English headers) |
| Rose 25-26 | `data/raw/25-26/Rose_fantalega-nicosia.xlsx` | All transactions of the 24-25 season auction (Italian headers on row 1) |
| Quotazioni 24-25 | `data/raw/24-25/Quotazioni_Fantacalcio_Stagione_2024_25.xlsx` | 24-25 season FVM (pairs with the 24-25 auction) |
| Quotazioni 23-24 | `data/raw/23-24/Quotazioni_Fantacalcio_Stagione_2023_24.xlsx` | 23-24 season FVM (pairs with the 23-24 auction) |
| output_rp 25-26 | `data/intermediate/25-26/output_rp.csv` | Stage 1 output ready (consumed by Stage 2B, not 2A) |

**Rose file semantics (important):** `data/raw/{SEASON}/Rose_*.xlsx` contains the most recent
completed auction *before* season {SEASON} starts — i.e. the auction reflecting the t-1 season's
performances. Season t-1 auction data is used to learn the FVM→price relationship for season t,
because the season-t auction does not exist yet at training time. Concretely for the 25-26 trial:
the file in `data/raw/25-26/` (24-25 season auction) is the primary (weight 1.0) training input,
and the file in `data/raw/24-25/` (23-24 season auction) is the secondary (weight 0.7) input.

---

## Architecture Target

```
Stage 2A: 02_fvm_to_distribution.ipynb
  Input:  all Rose_*.xlsx from data/raw/{SEASON}/ and prior seasons
          + per-season Quotazioni (FVM source: data/raw/{s}/Quotazioni_*.xlsx)
  Process:
    - Load + clean Rose files (handle Italian/English headers)
    - Tag each file with league metadata (credits, players, format, source)
    - Pair each auction with the FVM of its own season (Quotazioni)
    - Aggregate transactions per (player, auction) → mean price, count
    - Filter count >= MIN_COUNT
    - Per-role model training:
      * Mean: Ridge([sqrt(FVM), FVM]) — non-linear FVM→price
      * Std: Binned empirical std + Ridge smoothing — robust std estimation
    - Multi-season training with recency weights
    - Validation: hold-out, per-role MAE/RMSE, FVM-bin analysis
  Output: models/fvm_distribution/{SEASON}/ (tagged with league format + training seasons)

Stage 2B: 03_regressors.ipynb
  Input:  output_rp.csv + models from Stage 2A
  Process:
    - Load models, predict expprice/expstd for every player
    - Experimental: residual correction variants (ExpectedMf-based)
    - Validation: BE contract compliance, spot-check, distribution analysis
  Output: data/final/{SEASON}/players.csv (all 14 BE columns)
```

---

## Phase 3A — Notebook Standardization (Stage 2A)

### A1. First cell: SEASON + path helpers + config load
```python
import os, sys
from pathlib import Path

# Repo root detection
BASE = str(Path(__file__).parent.parent if '__file__' in globals() else Path.cwd().parent)
if not Path(BASE + '/pyproject.toml').exists():
    BASE = str(Path.cwd())

SEASON = "25-26"
SEASON_START_YEAR = 2025
RAW_DIR = f"data/raw/{SEASON}"
INTER_DIR = f"data/intermediate/{SEASON}"
MODELS_DIR = f"models/fvm_distribution/{SEASON}"
os.makedirs(INTER_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
```

### A2. Config YAML: `pipeline/config/stage2a2b.yaml`
```yaml
# Stage 2A - Price Model Training
training:
  min_count: 8                    # min appearances across leagues
  fvm_floor:                      # per-role FVM minimum for training
    P: 1
    D: 10
    C: 1
    A: 1
  recency_weight: 0.7             # weight for prior season Rose data (current = 1.0)
  league_format: "800_8"          # credits_players tag

# Stage 2A - Model configuration
model:
  mean_method: "ridge_sqrt"       # Ridge([sqrt(FVM), FVM])
  std_method: "binned_empirical"  # empirical std per FVM bin, smoothed
  ridge_alpha: 1.0                # Ridge regularization
  std_bin_width: 10               # FVM bin width for empirical std
  std_min: 1                      # floor for std predictions

# Stage 2B - Inference
inference:
  residual_correction: "none"     # none | additive | multiplicative | experimental
  residual_weight: 0.1            # strength of ExpectedMf-based correction
  price_min: 1                    # floor for expprice
  std_min: 1                      # floor for expstd

# Roles
roles:
  major: ["P", "D", "C", "A"]
```

### A3. Canonical auction input contract + multi-league loader

**Canonical record schema** — what every auction source must provide:

| Column | Type | Description |
|---|---|---|
| `role` | str | One of `P`, `D`, `C`, `A` |
| `name` | str | Player name, cleaned (trimmed, asterisks removed) |
| `team` | str | Team name |
| `price` | int | Auction price in credits |

**File conventions in `data/raw/{SEASON}/`** (folder = target season, per Rose semantics):
- `Rose_*.xlsx` — local multi-league auction exports (existing source). Header language auto-detected
  (Italian on row 1: `Ruolo/Calciatore/Squadra/Costo`; English: `Role/Name/Team/Price`).
- `auction_{league}_{format_tag}.csv` — external sources (e.g. website parser output, roadmap Phase 5),
  already in the canonical schema above. `format_tag = {credits}_{players}` (e.g. `800_8`),
  e.g. `auction_fantaclub_1000_10.csv`.

**Metadata** (derived at load time, not stored per record):
- `source_file`, `source` (`rose` | `website`), `league` (from filename)
- `auction_season` — the season the auction belongs to (a file in `data/raw/{t}/` is the t-1 season auction)
- `credits`, `players`, `format_tag` — from filename for `auction_*.csv`; default `800_8` for Rose xlsx (configurable in YAML)

**Loader behavior:**
- Glob `Rose_*.xlsx` + `auction_*.csv` from `data/raw/{SEASON}/` (and prior seasons for multi-season training)
- Auto-detect headers for xlsx; read canonical CSVs as-is
- Clean: drop rows with missing name/price, strip whitespace, remove asterisks, keep roles P/D/C/A only, coerce price to int
- Tag each record with its file metadata
- Return unified DataFrame: `role, name, team, price, source_file, source, league, format_tag`

**Why this contract now:** the 26-27 website study will emit `auction_*.csv` files conforming to this
schema; once validated they flow into Stage 2A training with zero loader changes. All current data is
`800_8`; per-format models are a config switch, not a refactor.

### A4. Multi-season training data assembly
- Load current Rose (`data/raw/{SEASON}/`, the most recent completed auction = t-1 season auction)
  + prior Rose (`data/raw/{SEASON-1}/`, if available)
- **Contemporaneous FVM pairing:** each auction is paired with the FVM of *its own season*,
  read from `data/raw/{s}/Quotazioni_*.xlsx` (header row auto-detected). The 24-25 auction
  uses the 24-25 FVM, the 23-24 auction the 23-24 FVM. Using the current-season FVM for
  older auctions would mix stale market values with current FVMs and destroy the
  FVM -> price relationship. At inference time the current-season FVM is fed in — exactly
  what the next season's auction will price.
- Apply recency weight: current Rose weight = 1.0, prior Rose weight = `recency_weight` (0.7)
- **Mathematical justification for recency weighting:**
  - Market perceptions evolve. A player valued at 80 last season may be valued at 100 this season due to improved performance or changing meta.
  - Weight `w_prior = 0.7` means prior season contributes ~70% of current season influence.
  - Weighted mean: `mean = sum(w_i * p_i) / sum(w_i)` where `w_i` depends on which season the price came from.
  - Weighted std: uses Welford's online algorithm for numerically stable weighted variance.

### A5. Price aggregation
- A Rose file contains every transaction of every fantasy team across all leagues of the
  export (same 20 team names per league), so the same real player appears many times —
  once per buying team, each at its own auction price
- Group by `(role, name, auction)` → mean price, transaction count
- Filter `count >= min_count` (8 transactions per player per auction = reliable market price)
- Apply per-role FVM floor
- Training unit: one row per `(player, auction)`, with its contemporaneous FVM and recency weight

---

## Phase 3B — Model Improvements (Stage 2A)

### B1. Non-linear mean model: `Ridge([sqrt(FVM), FVM])`

**Current problem:** `LinearRegression([FVM] → price)` forces a straight line. The FVM→price relationship is inherently non-linear:
- Low FVM (1-20): prices cluster near 1-5 (compressed)
- Mid FVM (20-100): prices grow roughly linearly (5-50)
- High FVM (100-400+): prices accelerate (50-300+)

**Proposed model:** `Ridge(alpha=1.0)` on features `[sqrt(FVM), FVM]`

**Mathematical justification:**
- Feature `sqrt(FVM)` captures the sub-linear growth at low FVM (compresses the 1-20 range)
- Feature `FVM` captures the linear/exponential growth at high FVM
- Ridge regularization (`alpha=1.0`) prevents multicollinearity between `sqrt(FVM)` and `FVM`
- The model learns: `price ≈ β0 + β1*sqrt(FVM) + β2*FVM`
- With `β1 > 0, β2 > 0`: sub-linear at low FVM, super-linear at high FVM — matching observed data
- This is equivalent to a quadratic in `sqrt(FVM)`: `price ≈ β0 + β1*x + β2*x²` where `x = sqrt(FVM)`
- The quadratic form is monotonically increasing for `x >= -β1/(2*β2)`, which holds since FVM >= 1

**Validation:** Compare MAE/RMSE vs baseline LinearRegression([FVM]) per role. Accept if MAE decreases in >= 3/4 roles.

### B2. Robust std model: Binned empirical std + Ridge smoothing

**Current problem:** `LinearRegression([FVM] → std)` is conceptually weak:
- Std doesn't have a clean linear relationship with FVM
- Low FVM players: low std (consensus on low value)
- Mid FVM: moderate std
- High FVM: can be high std (disagreement on star valuations)
- The relationship may be inverted-U or saturating, not linear

**Proposed approach:** Two-stage estimation
1. **Binned empirical std:** Bin players by FVM (bin width = 10). For each bin, compute the empirical std of prices. This gives a non-parametric estimate of `std(FVM)`.
2. **Ridge smoothing:** Fit `Ridge([FVM] → empirical_std)` on the bin centers. This smooths noisy bins while preserving the non-linear shape.

**Mathematical justification:**
- Binning is a non-parametric estimator: no assumption about functional form
- Bin width 10 balances resolution (20-40 bins across FVM range 1-400) vs sample size per bin
- Ridge smoothing on bin centers prevents overfitting to small bins
- Floor at `std_min = 1` ensures all players have non-zero uncertainty
- For FVM values outside training range, extrapolation uses the Ridge model (linear extrapolation of smoothed curve)

**Validation:** Compare predicted std vs actual std per FVM bin. Accept if mean absolute error of std < 3 credits.

### B3. Per-role model training
- Train separate models per role (P, D, C, A)
- Each role has different FVM→price dynamics (e.g., defenders have compressed price range, attackers have wider spread)
- Apply per-role FVM floor before training

### B4. Training visualization
- Scatter plot: FVM vs price per role, with Ridge fit overlay
- Residual plot: predicted - actual vs FVM (check for patterns)
- Std plot: empirical std per FVM bin, with smoothed curve overlay
- Correlation table: per-role FVM-price correlation

---

## Phase 3C — Validation Framework (Stage 2A)

### C1. Hold-out validation
- Split Rose teams: 80% training, 20% hold-out
- Train models on 80%, predict on 20%
- Metrics: MAE, RMSE per role for mean price

### C2. Per-role MAE/RMSE
```
Role P: MAE=..., RMSE=..., n=...
Role D: MAE=..., RMSE=..., n=...
Role C: MAE=..., RMSE=..., n=...
Role A: MAE=..., RMSE=..., n=...
```

### C3. FVM-bin analysis
- Bin players by FVM: [1-10], [11-20], [21-50], [51-100], [101-200], [200+]
- For each bin: report MAE, count, mean_actual, mean_predicted
- Identifies where model over/under-predicts

### C4. Cross-season drift check
- Both Rose files are available (23-24 auction in `data/raw/24-25/`, 24-25 auction in `data/raw/25-26/`):
  - Train on the 24-25 Rose only (23-24 auction), predict the 25-26 Rose prices (24-25 auction)
    → error quantifies market drift between consecutive auctions
  - Compare against the in-auction hold-out MAE (C1) to separate drift from model error

### C5. Validation JSON output
```json
{
  "season": "25-26",
  "training_seasons": ["24-25", "25-26"],
  "recency_weight": 0.7,
  "model_mean": "ridge_sqrt",
  "model_std": "binned_empirical",
  "hold_out_split": 0.2,
  "per_role_metrics": {
    "P": {"mae": ..., "rmse": ..., "n_train": ..., "n_holdout": ...},
    "D": {...},
    "C": {...},
    "A": {...}
  },
  "fvm_bin_metrics": {
    "1-10": {"mae": ..., "count": ...},
    ...
  },
  "cross_season_drift": {
    "mae_24_25_only": ...,
    "mae_25_26_only": ...
  }
}
```

---

## Phase 3D — Stage 2B Refactoring

### D1. Notebook standardization
- SEASON variable, repo-root detection, config load (same pattern as Stage 2A)
- Load models from `models/fvm_distribution/{SEASON}/`
- Load `output_rp.csv` from `data/intermediate/{SEASON}/`

### D2. Production inference (cells 0-10)
- For each player, predict expprice and expstd using role-specific models
- Apply floors: `expprice >= price_min`, `expstd >= std_min`
- Round to integers (BE expects int for expprice, expstd)

### D3. Residual correction experiments (cells 11+)

**Motivation:** Players whose ExpectedMf exceeds their role's median for their FVM band may be undervalued by the FVM-only model. The residual correction adjusts for this.

**Experimental variants to test:**

**Variant 0 — Baseline (no correction):**
```
expprice_final = expprice_from_model
```

**Variant 1 — Additive correction:**
```
role_median_mf = median(ExpectedMf for players in same role and FVM bin)
mf_deviation = ExpectedMf - role_median_mf
expprice_final = expprice_from_model + residual_weight * mf_deviation * 10
```
- `residual_weight * 10` scales the correction to credit units (ExpectedMf is in 1-10 scale, credits are in 1-300+ scale)
- Mathematical justification: if a player's expected performance is 0.5 above role median, and we weight at 0.1, the correction is `0.1 * 0.5 * 10 = 0.5` credits

**Variant 2 — Multiplicative correction:**
```
role_median_mf = median(ExpectedMf for players in same role and FVM bin)
mf_ratio = ExpectedMf / role_median_mf
expprice_final = expprice_from_model * (1 + residual_weight * (mf_ratio - 1))
```
- Mathematical justification: proportional adjustment. A player 10% above median gets a `residual_weight * 0.1` proportional boost.

**Variant 3 — Capped additive:**
```
correction = clip(residual_weight * mf_deviation * 10, -cap, +cap)
expprice_final = expprice_from_model + correction
```
- Cap prevents extreme adjustments (e.g., cap = 5 credits)

**Validation approach:**
- For each variant, compute predicted prices for players with known auction prices (from Rose data)
- Compare MAE vs baseline (no correction)
- Visual inspection: do top players look more realistic?
- User's eye has final say — the quantitative metrics inform but don't decide

### D4. Final CSV export
- Rename columns to BE contract names
- Save to `data/final/{SEASON}/players.csv`
- Validation: schema check, no duplicates, positive values

### D5. Archive experimental cells
- Move cells 12-45 (GPR, Ridge, Lasso, SVM experiments) to separate exploration notebook or collapse into a single "Archived experiments" section at the end
- These are not part of production flow

---

## Phase 3E — Housekeeping

### E1. Archive `04_expected_price.ipynb`
- Move to `pipeline/archive/04_expected_price.ipynb`
- Add markdown cell at top: "ARCHIVED — old multi-feature Lasso path. Not part of active pipeline. Replaced by Stage 2A (Ridge sqrt) + Stage 2B (residual correction)."

### E2. Config centralization
- `pipeline/config/stage2a2b.yaml` — all Stage 2A/2B parameters
- Follows Stage 01 pattern (single YAML, loaded at notebook start)

### E3. Docs update
- Update `docs/improvement-roadmap.md` — mark Phase 3 items as done
- Update `docs/ml-be-contract.md` (already done for Sections 5-7)
- Update notebook markdown cells to match actual methodology

---

## Prerequisite P1 — Restore MyRating in Stage 1 output (BE contract)
The FE displays the rating, so `myrating` must be present in the final CSV. It was auto-computed in the
old ratings notebook and dropped in the Stage 01 refactor. Restore it as a small Stage 1 addition:

- Per role, compute pooled historical Mf mean/std (seasons with stats, Pg >= age_min_games)
- Closed form, old absolute scale, no curve fitting:
  ```
  z = (ExpectedMf - mean_role) / std_role
  MyRating = round(clip(1 + (z + 4) / 6.5 * 4, 1, 5), 1)    # z in [-4, 2.5] -> [1, 5]
  ```
- Deterministic, no manual step. `output_rp.csv` gains the `MyRating` column again; Stage 2B passes it through unchanged.

## Progress
* [x] Step 0 — P1: MyRating restored in Stage 1. 25-26 output verified: 2.8–5.0, mean 3.57, 0 nulls
      (legacy 24-25 scale: 2.4–5.0, mean 3.44); validation JSON now checks it for real.
* [x] Step 1 — A1-A3: `02_fvm_to_distribution.ipynb` standardized (SEASON, repo-root, config),
      `pipeline/auction_loader.py` + `pipeline/config/stage2a2b.yaml` created. Smoke test: both
      auctions load (5,600 records, 20 teams each), IT/EN header auto-detection works, invariants pass.
* [x] Step 2 — A4-A5: **design fix vs the original plan.** The plan merged the current-season FVM
      (from `output_rp.csv`) into all auctions — this mixes stale market values with current FVMs
      (FVM-price r collapsed to 0.40–0.67, hold-out MAE 27–64). Each auction is now paired with the
      FVM of *its own season* (`data/raw/{s}/Quotazioni_*.xlsx`, header auto-detected; the 23-24
      Quotazioni was moved from repo root to `data/raw/23-24/`). Rose files contain every transaction
      of every fantasy team across all leagues of the export, so prices aggregate per
      `(role, name, auction)` and `min_count` keeps its legacy semantics (>= 8 transactions).
      Result: 374 training points, FVM-price Pearson r 0.91–0.96.
* [x] Step 3 — B1: Ridge sqrt beats the legacy baseline in 3/4 roles (P 6.22<7.45, D 5.68<5.70,
      A 16.55<17.80; C tie 8.83≈8.80). Acceptance met (>= 3/4).
* [x] Step 4 — B2: binned empirical std + Ridge smoothing. Std MAE vs empirical bins: P 4.9, D 5.4,
      C 9.2, A 23.9 (absolute credits). The plan's "< 3 credits" threshold was written for the
      legacy std scale (max 35); on current data stds reach 70+, so relative error is moderate.
      Flagged for review — not auto-passed.
* [x] Step 5 — B3-B4: per-role training, scatter+fit+1-std-band plots, std-bin plots, correlation table
* [x] Step 6 — C1-C5: hold-out MAE P 11.6 / D 5.4 / C 9.3 / A 22.6; drift 23-24→24-25 MAE 10.9 (n=172);
      `validation_stage2a.json` written. Overfit check: D/C pass; P (n=9) and A (n=19) hold-out
      samples too small to be conclusive — the drift check corroborates A's error level (22.4).
* [x] Step 7 — D1-D2: `03_regressors.ipynb` standardized; expprice/expstd predicted for all 532 players
* [x] Step 8 — D3: residual correction variants vs observed 24-25 auction prices (n=207): no variant
      beats baseline (MAE 21.64–21.65) — the correction is too weak relative to cross-season drift.
      Production stays `residual_correction: none`.
* [x] Step 9 — D4-D5: `data/final/25-26/players.csv` (532×14, all BE checks pass); legacy experiment
      cells (old 03 cells 12-45) moved to `explorations/legacy_regressor_experiments.ipynb`
* [x] Step 10 — E1-E3: `04_expected_price.ipynb` archived to `pipeline/archive/`; docs updated

**Pending user review:** top-20 price plausibility (Stage 2B spot-check), std-model MAE threshold,
and the P/A overfit-check sample sizes.

## Implementation Order

| Step | Phase | Description | Validation |
|---|---|---|---|
| 0 | P1 | Restore MyRating in Stage 1 (auto-derived) | `myrating` in output_rp.csv, 1–5 range, 1 decimal |
| 1 | A1-A3 | Standardize notebook, canonical auction loader (A3 contract), config | Rose + canonical CSV load correctly, headers auto-detected |
| 2 | A4-A5 | Multi-season training, price aggregation | Weighted mean/std computed, FVM merge clean |
| 3 | B1 | Ridge([sqrt(FVM), FVM]) mean model | MAE vs baseline LinearRegression per role |
| 4 | B2 | Binned empirical std model | Std MAE < 3 credits per FVM bin |
| 5 | B3-B4 | Per-role training + visualizations | Plots show reasonable fits |
| 6 | C1-C5 | Full validation framework | JSON report, hold-out metrics |
| 7 | D1-D2 | Stage 2B standardization + inference | BE contract holds, all 14 columns |
| 8 | D3 | Residual correction experiments | MAE comparison, visual inspection |
| 9 | D4-D5 | Final export + archive experiments | Clean output, no errors |
| 10 | E1-E3 | Housekeeping + docs | Roadmap updated, archive clean |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Ridge sqrt overfits small roles (P has few players) | Ridge alpha regularization; hold-out validation catches overfit |
| Binned std has empty bins at extreme FVM | Merge adjacent bins if count < 3; extrapolate via Ridge |
| Multi-season training introduces stale data | Recency weight 0.7 down-weights prior season; cross-season drift check |
| Residual correction over-adjusts | Capped variant limits max adjustment; experimental session compares all variants |
| Rose file format changes | Auto-detect headers (Italian/English); metadata schema is extensible |
| 25-26 Rose data not final | Flag in config; retrain when final data available |

---

## Acceptance Criteria

### Stage 2A
- [x] Notebook runs end-to-end for SEASON="25-26" without errors
- [x] Ridge sqrt MAE < LinearRegression MAE in >= 3/4 roles (P, D, A; C tie)
- [ ] Std MAE < 3 credits across FVM bins — **not met as written** (4.9–23.9 absolute; threshold
      is legacy-scale, see Progress step 4); needs review / re-thresholding
- [ ] Hold-out validation shows no overfitting (holdout MAE < 1.2 * training MAE) — D/C pass;
      P (n=9) / A (n=19) inconclusive due to sample size; drift check corroborates A
- [x] Validation JSON written to `data/intermediate/{SEASON}/validation_stage2a.json`
- [x] Visualizations present: scatter + fit, std bins, correlation table

### Stage 2B
- [x] Notebook runs end-to-end, produces `data/final/{SEASON}/players.csv`
- [x] All 14 BE columns present, correctly named
- [x] expprice > 0, expstd > 0 for all players
- [x] No duplicate Id
- [x] Residual correction experiment cell present with variant comparison
- [ ] Top 20 players have plausible prices (visual check — pending user review)

### Overall
- [x] Config in `pipeline/config/stage2a2b.yaml`
- [x] `04_expected_price.ipynb` archived in `pipeline/archive/`
- [x] Roadmap updated
- [x] Notebook markdown cells match methodology
