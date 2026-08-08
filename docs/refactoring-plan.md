# FantaHelpML Refactoring Plan

## 1. Current State

### 1.1 Repository Layout (As-Is)

The repo is a flat collection of notebooks, data files, and model artifacts with three overlapping season directories and no clear separation between pipeline stages.

```
FantaHelpML/
├── data_preprocess_merge.ipynb       # Stage 0: Data ingestion & merge
├── ratings.ipynb                     # Stage 1: Performance curves & ratings
├── fvm_to_distribution.ipynb         # Stage 2A: Train LinearRegression from real auction data
├── regressors.ipynb                  # Stage 2B: Apply models + exploratory training
├── expected_price.ipynb              # Stage 3: Predecessor pipeline (23-24 season, MultiOutput Lasso)
├── utility.py                        # Utility functions
├── 24-25/                            # Current season data
├── 24-25oldge/                       # Alternate run (older quotation edition, includes SQLite export)
├── 24-25_trial/                      # Earlier trial run with older FVM data
├── regressors/24-25/                 # Joblib models (mean + std per role, trained by Stage 2A)
├── stats/                            # Historical stats 2019-2024
├── data/utils/                       # xlsx_to_pkl.py
├── explorations/                     # expected_value_players.ipynb (post-hoc validation)
├── [various CSVs, Excel, pkl in root] # Clutter
```

### 1.2 Pipeline Flow (As-Is)

The pipeline is a **sequential chain** of 5 stages. The active production path is Stages 0 → 1 → 2A → 2B. Stage 3 is the predecessor pipeline from the 23-24 season (replaced by 2A+2B for 24-25). Stage 4 is post-hoc validation.

```
Stage 0: data_preprocess_merge.ipynb
  Input:  Quotazioni_Fantacalcio_Stagione_*.xlsx, stats/ (5 seasons), previous season merged data
  Output: data_preprocess_merge.xlsx
  NOTE:    Manual intervention required after this step (update Age, Regularness, Mate)

Stage 1: ratings.ipynb
  Input:  data_preprocess_merge.xlsx
  Output: output_rp.csv (columns: Id, Role, Role_M, Name, Squad, Price, Age, MyRating,
          Mate, Regularness, FVM, ExpectedMf, historical stats)

Stage 2A: fvm_to_distribution.ipynb
  Input:  output_rating.csv (from Stage 1 — should be output_rp.csv, see Fix 4.3)
          Rose_fantalega-nicosia.xlsx (real auction data)
  Output: regressors/model_[mean,std]_[P,D,C,A].joblib (8 LinearRegression models)
  NOTE:    Trains models on empirical auction prices (count >= 8 filter)

Stage 2B: regressors.ipynb
  Input:  output_rp.csv (from Stage 1)
          regressors/24-25/model_[mean,std]_[P,D,C,A].joblib (from Stage 2A)
  Output: players23_24_nostats.csv (BE-ready, columns renamed to match BE contract)
  NOTE:    Production cells (0-11) apply models. Exploratory cells (12-45) train .pkl models.
          **This notebook produces the active final output for the BE.**

Stage 3: expected_price.ipynb  (PREDECESSOR, 23-24 season)
  Input:  output_rating.csv (root-level, older format)
          squads.csv
          lasso_regressor_model_*.pkl (trained by regressors.ipynb exploratory cells)
  Output: players23_24_trial.csv, players23_24_nostats.csv
  NOTE:    Was the production path for 23-24 season. Replaced by Stages 2A+2B for 24-25.
          Uses MultiOutput Lasso (5 features). Root has lasso A+C only; full set in 24-25_trial/.
          Ridge models (A, C, D, P) exist at root.

Stage 4: explorations/expected_value_players.ipynb  (post-hoc validation)
  Input:  output_rating.csv, Rose_*.xlsx
  Output: player_prices.xlsx, friends_prices.xlsx
  NOTE:    Empirical inter-league price comparison. Same approach as Stage 2A but
          computes per-player stats instead of training models.
```

### 1.3 Issues Identified

| # | Issue | Severity | Detail |
|---|---|---|---|
| 1 | **Age and Role_M dropped from final output** | HIGH | `regressors.ipynb` `columns_to_keep` does not include `Age` or `Role_M`. BE entity has both fields but DTO doesn't import them. Fix: add to columns_to_keep + rename_dict, add to PlayerCreateDto. |
| 2 | **Stale output filename** | MEDIUM | `regressors.ipynb` writes `players23_24_nostats.csv` regardless of season. Should use dynamic naming (e.g., `players24_25.csv`). |
| 3 | **Scattered model files** | HIGH | Lasso models in root (`lasso_regressor_model_*.pkl`), in `24-25_trial/`, and ridge models also in root. Joblib models in `regressors/24-25/`. No single source of truth. |
| 4 | **Duplicate season dirs** | HIGH | `24-25`, `24-25oldge`, `24-25_trial` contain overlapping data with different column layouts. Unclear which is canonical. |
| 5 | **fvm_to_distribution.ipynb not in pipeline flow** | MEDIUM | This notebook trains the production LinearRegression models but is not clearly integrated into the pipeline order. It reads from `output_rating.csv` (older name) instead of `output_rp.csv`. |
| 6 | **No pipeline documentation** | MEDIUM | README.md only lists step names. No input/output contracts, no data flow diagram, no season transition guide. |
| 7 | **Hardcoded paths** | MEDIUM | Notebooks reference files by relative names (`output_rating.csv`, `squads.csv`) without documenting where they should live. |
| 8 | **Root clutter** | LOW | CSVs, Excel files, pkl files scattered in root. |
| 9 | **Season coupling** | LOW | `current_season = "24-25"` hardcoded in notebooks. No abstraction for season switching. |
| 10 | **Uncommitted work risk** | HIGH | Work on remote laptop (1500km away, 3 weeks). No safe integration strategy. |
| 11 | **Manual intervention required** | INFO | After Stage 0, `Age`, `Regularness`, and `Mate` must be manually updated. This is by design but should be documented clearly. |
| 12 | **Stale squads.csv** | LOW | `squads.csv` contains 2020-21 teams (BENEVENTO, CROTONE, PARMA, SPEZIA — all relegated). **Only affects predecessor pipeline (Stage 3/expected_price.ipynb).** Not used by active Stages 0-2B. Must be updated annually if Stage 3 is reactivated. |
| 13 | **Magic number thresholds** | MEDIUM | `ratings.ipynb` uses scattered undocumented thresholds: `min_matches=11`, per-role `matches_filter=[8,25,22,29,29,27,26,28,24,19,27,29]`, FVM curve requires `>=22` games, age-prediction branch uses `>=16` games. Should be parameterized. |
| 14 | **Bug in xlsx_to_pkl.py** | LOW | `main()` calls `process_teams_data(excel_file)` which doesn't exist — should be `convert_xlsx_data_to_pkl(excel_file)`. |
| 15 | **Dual output from ratings.ipynb** | LOW | Produces both `output_rp.csv` and `output_rp_excel.xlsx`. Only the CSV is canonical; the Excel output could cause confusion. |

### 1.4 Data Flow Analysis

**What each notebook produces vs what the BE needs:**

| Notebook | Output Columns | BE Needs | Gap |
|---|---|---|---|
| `ratings.ipynb` | Id, Role, Role_M, Name, Squad, Price, Age, MyRating, Mate, Regularness, FVM, ExpectedMf, [historical stats] | All except historical stats | OK |
| `fvm_to_distribution.ipynb` | Trains .joblib models (no CSV output) | N/A | Models are consumed by Stage 2B |
| `regressors.ipynb` | id, role, name, squad, price, myRating, mate, regularness, fvm, expMf, expPrice, expStd | Same + age, role_m | **Age and Role_M present in input but dropped by columns_to_keep** |
| `expected_price.ipynb` | ExpectedPrice, ExpectedPriceStd (on different data branch) | N/A | Alternative path, not part of active pipeline |

**The BE needs a single CSV with:**
```
id, role, name, squad, price, myrating, mate, regularness, fvm, expmf, expprice, expstd, age, role_m
```

**`regressors.ipynb` produces this CSV** (via rename_dict), but `columns_to_keep` drops `Age` and `Role_M` before the rename step. The fix is to add these two columns to `columns_to_keep` and their lowercase mappings to `rename_dict`. No pipeline reconnection is needed.

---

## 2. Target Structure

### 2.1 Directory Layout

```
FantaHelpML/
├── README.md                       # Pipeline overview, run instructions, annual guide
├── docs/
│   └── refactoring-plan.md         # This document
├── pipeline/                       # Ordered pipeline notebooks
│   ├── 00_data_preprocess_merge.ipynb   # Stage 0: Data ingestion & merge
│   ├── 01_ratings.ipynb                   # Stage 1: Performance curves & ratings
│   ├── 02_fvm_to_distribution.ipynb       # Stage 2A: Train LinearRegression from auction data
│   ├── 03_regressors.ipynb                # Stage 2B: Apply models + exploratory training
│   └── 04_expected_price.ipynb            # Predecessor pipeline (23-24 season, MultiOutput Lasso)
├── data/
│   ├── raw/                        # Source data per season (never modified)
│   │   └── 24-25/
│   │       ├── Quotazioni_Fantacalcio_Stagione_2024_25.xlsx
│   │       ├── Rose_*.xlsx
│   │       ├── squads.csv
│   │       └── copyCsvReal.xlsx
│   ├── intermediate/               # Pipeline intermediate outputs per season
│   │   └── 24-25/
│   │       ├── data_preprocess_merge.xlsx  # Stage 0 output
│   │       └── output_rp.csv               # Stage 1 output
│   ├── final/                      # BE-ready CSV per season
│   │   └── 24-25/
│   │       └── players.csv         # Single file to import into BE (from Stage 2B)
│   ├── historical/                 # Previous seasons stats
│   │   ├── Statistiche_Fantacalcio_Stagione_2019_20.xlsx
│   │   ├── Statistiche_Fantacalcio_Stagione_2020_21.xlsx
│   │   ├── Statistiche_Fantacalcio_Stagione_2021_22.xlsx
│   │   ├── Statistiche_Fantacalcio_Stagione_2022_23.xlsx
│   │   └── Statistiche_Fantacalcio_Stagione_2023_24.xlsx
│   └── utils/                      # Utility scripts
│       └── xlsx_to_pkl.py
├── models/                         # Trained models
│   ├── fvm_distribution/           # Joblib models (FVM → price mean/std, trained by Stage 2A)
│   │   └── 24-25/
│   │       ├── model_mean_P.joblib
│   │       ├── model_mean_D.joblib
│   │       ├── model_mean_C.joblib
│   │       ├── model_mean_A.joblib
│   │       ├── model_std_P.joblib
│   │       ├── model_std_D.joblib
│   │       ├── model_std_C.joblib
│   │       └── model_std_A.joblib
│   └── multi_feature/              # Lasso/ridge models (5-feature, from Stage 2B experiments)
│       ├── lasso_P.pkl
│       ├── lasso_D.pkl
│       ├── lasso_C.pkl
│       ├── lasso_A.pkl
│       ├── ridge_P.pkl
│       ├── ridge_D.pkl
│       ├── ridge_C.pkl
│       └── ridge_A.pkl
├── explorations/                   # Experimental notebooks (not in pipeline)
│   └── expected_value_players.ipynb   # Post-hoc inter-league price comparison
└── scratch/                        # Landing zone for uncommitted remote work
    └── .gitkeep
```

### 2.2 Pipeline Contract

Each notebook reads from and writes to well-defined paths relative to a season directory:

```
00_data_preprocess_merge.ipynb
  Reads:  data/raw/{season}/Quotazioni_*.xlsx
          data/raw/{season}/copyCsvReal.xlsx
          data/historical/
  Writes: data/intermediate/{season}/data_preprocess_merge.xlsx
  NOTE:    Manual intervention required after (update Age, Regularness, Mate)

01_ratings.ipynb
  Reads:  data/intermediate/{season}/data_preprocess_merge.xlsx
  Writes: data/intermediate/{season}/output_rp.csv

02_fvm_to_distribution.ipynb
  Reads:  data/intermediate/{season}/output_rp.csv
          data/raw/{season}/Rose_*.xlsx
  Writes: models/fvm_distribution/{season}/model_[mean,std]_[P,D,C,A].joblib

03_regressors.ipynb  (ACTIVE FINAL OUTPUT)
  Reads:  data/intermediate/{season}/output_rp.csv
          models/fvm_distribution/{season}/
  Writes: data/final/{season}/players.csv

04_expected_price.ipynb  (PREDECESSOR, 23-24 season)
  Reads:  data/intermediate/{season}/output_rp.csv
          models/multi_feature/
          data/raw/{season}/squads.csv  [UPDATE ANNUALLY — currently stale]
  Writes: data/final/{season}/players_lasso.csv  (predecessor output)
```

### 2.3 Final Output Format

`data/final/{season}/players.csv` -- the single file imported into the BE. Column names match BE's `PlayerCreateDto` (case-insensitive CsvHelper match):

```
id, role, name, squad, price, myrating, mate, regularness, fvm, expmf, expprice, expstd, age, role_m
```

---

## 3. Migration Steps

### Phase 1: Safety (no content changes)

**Pre-step:** Create `git tag pre-refactoring` for rollback capability.

1. Create new directory structure (empty dirs)
2. Move `stats/` → `data/historical/`
3. Move `data/utils/` → stays at `data/utils/` (already correct)
4. Move `explorations/` → stays (already correct)
5. Create `scratch/` with `.gitkeep`
**Verify:** Git diff shows only renames/moves, no content changes.

### Phase 2: Consolidate season data

**Canonical source identification:**
- `24-25/` = latest pipeline run (canonical for raw data and intermediate outputs)
- `24-25_trial/` = earlier trial with older FVM data + full set of trained models (archive)
- `24-25oldge/` = alternate quotation edition + SQLite export (archive)

6. Move raw source files from `24-25/` → `data/raw/24-25/`:
   - `Quotazioni_Fantacalcio_Stagione_2024_25.xlsx`
   - `Rose_fantalega-nicosia.xlsx`
   - `squads.csv` (stale 2020-21 data — only affects predecessor pipeline, not active path)
   - `copyCsvReal.xlsx`
7. Move intermediate outputs from `24-25/` → `data/intermediate/24-25/`:
   - `data_preprocess_merge.xlsx`
   - `output_rp.csv`
8. Move final output from root → `data/final/24-25/players.csv`
9. Archive `24-25_trial/` and `24-25oldge/` contents (preserve models, remove dupes)
**Verify:** All file paths resolve. No broken references.

### Phase 3: Consolidate models

10. Move joblib models from `regressors/24-25/` → `models/fvm_distribution/24-25/`
11. Move lasso/ridge pkl models from root → `models/multi_feature/`
12. Copy missing lasso models from `24-25_trial/` (`lasso_D.pkl`, `lasso_P.pkl`) → `models/multi_feature/`
13. Remove duplicate pkl files from `24-25_trial/` after verification
**Verify:** All model files load correctly (test with `joblib.load()`).

### Phase 4: Pipeline notebooks

14. Move `data_preprocess_merge.ipynb` → `pipeline/00_data_preprocess_merge.ipynb`
15. Move `ratings.ipynb` → `pipeline/01_ratings.ipynb`
16. Move `fvm_to_distribution.ipynb` → `pipeline/02_fvm_to_distribution.ipynb`
17. Move `regressors.ipynb` → `pipeline/03_regressors.ipynb`
18. Move `expected_price.ipynb` → `pipeline/04_expected_price.ipynb`
19. Update all file paths in notebooks to use new structure
20. Fix `columns_to_keep` in regressors to include `Age` and `Role_M`
21. Fix `rename_dict` to include `Age` → `age` and `Role_M` → `role_m`
22. Update `fvm_to_distribution.ipynb` to read `output_rp.csv` instead of `output_rating.csv`
23. Ensure final output filename uses season-aware naming
24. Fix bug in `xlsx_to_pkl.py`: rename `process_teams_data` → `convert_xlsx_data_to_pkl`
**Verify:** Run pipeline end-to-end with new paths. Confirm final CSV matches BE contract.

### Phase 5: Documentation

25. Write `README.md` with pipeline flow, run instructions, annual guide
26. Update `docs/refactoring-plan.md` with completion notes

---

## 4. Pipeline Fixes Required

### 4.1 Include Age and Role_M in Final Output

**Current problem:** `regressors.ipynb` `columns_to_keep` drops `Age` and `Role_M` even though they are present in the input (`output_rp.csv`).

**Fix in `regressors.ipynb`:**

```python
# Current:
columns_to_keep = ["Id", "Role", "Name", "Squad", "Price", "MyRating", "Mate", "Regularness", "FVM", "ExpectedMf", "mean", "std"]

# Fixed:
columns_to_keep = ["Id", "Role", "Role_M", "Name", "Squad", "Price", "Age", "MyRating", "Mate", "Regularness", "FVM", "ExpectedMf", "mean", "std"]
```

Add to rename_dict:
```python
"Age": "age",
"Role_M": "role_m",
```

**Fix in BE (`PlayerCreateDto`):** Add `Age` (int) and `Role_M` (string) properties.

**Fix in BE (`PlayerService`):** Parse Role_M with `Split(';')` instead of stubbing as `[Role]`.

### 4.2 Dynamic Output Filename

**Current problem:** `regressors.ipynb` writes `players23_24_nostats.csv` regardless of season.

**Fix:** Use season-aware naming:
```python
dataframe_merge.to_csv(f'players{SEASON.replace("-", "_")}.csv', index=False, sep=',', encoding='utf-8')
```

### 4.3 Unify Input File Names

**Current problem:** `fvm_to_distribution.ipynb` reads `output_rating.csv` while `ratings.ipynb` writes `output_rp.csv`. These should be the same file.

**Fix:** Update `fvm_to_distribution.ipynb` to read `output_rp.csv` (the canonical Stage 1 output).

### 4.4 Season Abstraction

**Current problem:** `current_season = "24-25"` hardcoded in notebooks. File paths are relative and undocumented.

**Fix:** First cell of each pipeline notebook defines the season variable. Path helpers construct file paths:

```python
SEASON = "24-25"
RAW_DIR = f"data/raw/{SEASON}"
INTER_DIR = f"data/intermediate/{SEASON}"
FINAL_DIR = f"data/final/{SEASON}"
MODELS_DIR = f"models/fvm_distribution/{SEASON}"
```

---

## 5. Integration Strategy for Uncommitted Work

### 5.1 `scratch/` Directory

Purpose: safe landing zone for files from the remote laptop when it becomes available again.

Process:
1. Copy uncommitted/new files from remote laptop into `scratch/`
2. Compare with pipeline notebooks to identify improvements
3. Cherry-pick good changes into `pipeline/` notebooks
4. Move new data files into proper `data/` subdirectories
5. Delete `scratch/` contents when integrated

### 5.2 Multi-Season Analysis Preparation

The `data/historical/` directory already has 5 seasons of stats. The pipeline notebooks should be designed to accept historical data as an optional input for improving predictions. This is a future enhancement, not a refactoring concern, but the directory structure supports it.

---

## 6. Annual Season Transition Guide

Every year, the workflow is:

0. **Update historical stats** → add the just-completed season's stats Excel to `data/historical/`
1. **Collect new season raw data** → place in `data/raw/{new-season}/`:
   - `Quotazioni_Fantacalcio_Stagione_*.xlsx` (official SkySport quotations)
   - `Rose_*.xlsx` (real league auction data from prior season or early auctions)
   - `squads.csv` (team strength ratings — **must update annually** for current Serie A teams)
   - `copyCsvReal.xlsx` (previous season merged data, for Age carry-forward)
2. **Update season variable** in pipeline notebooks (first cell of each: `SEASON = "xx-xy"`)
3. **Run pipeline:**
   - `00_data_preprocess_merge.ipynb` → produces `data_preprocess_merge.xlsx`
   - **Manual step:** Update `Age`, `Regularness`, `Mate` fields
   - `01_ratings.ipynb` → produces `output_rp.csv` (ExpectedMf + MyRating)
   - `02_fvm_to_distribution.ipynb` → trains LinearRegression models → `.joblib` files
   - `03_regressors.ipynb` → applies models → produces final `players.csv`
4. **Validate output:** Spot-check expPrice, expStd, expMf values for reasonableness
5. **(Optional) Run exploration notebook** → compare model predictions against empirical auction data
6. **Import into BE:** `data/final/{new-season}/players.csv` → `POST /api/players/import`
7. **Archive previous season** (data stays in place, no cleanup needed)

---

## 7. Changelog

| Date | Action | Status |
|------|--------|--------|
| 2026-08-06 | Plan created | Draft |
| 2026-08-07 | Major rewrite after full pipeline discovery | Revised |
| | - Discovered `fvm_to_distribution.ipynb` as core Stage 2A (model training from real auction data) | |
| | - Corrected pipeline: sequential chain, not disconnected | |
| | - `expected_price.ipynb` reclassified as alternative path (not broken continuation) | |
| | - Removed "Disconnected pipeline" BLOCKING issue | |
| | - Added `fvm_to_distribution.ipynb` to pipeline/ (not explorations/) | |
| | - Primary gap identified: Age and Role_M dropped by columns_to_keep | |
| | - Added manual intervention note after Stage 0 | |
| 2026-08-07 | Review corrections applied | Revised v2 |
| | - `expected_price.ipynb` clarified as predecessor pipeline (23-24), not just "alternative" | |
| | - Fixed model inventory: root has lasso A+C + ridge A,C,D,P; full lasso set in 24-25_trial/ | |
| | - Removed squads.csv from ratings.ipynb inputs (it doesn't use it) | |
| | - Added Issue #12: Stale squads.csv (2020-21 teams, HIGH) | |
| | - Added Issue #13: Magic number thresholds in ratings.ipynb (MEDIUM) | |
| | - Added Issue #14: Bug in xlsx_to_pkl.py (LOW) | |
| | - Added Issue #15: Dual output from ratings.ipynb (LOW) | |
| | - Added rollback strategy (git tag) and per-phase verification steps | |
| | - Added canonical file identification for Phase 2 | |
| | - Added annual squads.csv update and historical stats update to annual guide | |
| 2026-08-07 | Re-review corrections applied | Revised v3 |
| | - Fixed target layout comment: `04_expected_price.ipynb` now says "Predecessor pipeline" | |
| | - Fixed step numbering: sequential 1-26 across all phases (was 1-9, 11-14, 14-23,26, 24-25) | |
| | - Added parenthetical to Stage 2A input in §1.2 pointing to Fix 4.3 | |
| | - Downgraded Issue #12 (squads.csv) from HIGH to LOW (only affects predecessor pipeline) | |
| | - Updated Phase 2 note to reflect squads.csv only affects predecessor path | |
| 2026-08-07 | Refactoring plan executed & verified | Completed |
| | - Git tag `pre-refactoring` created | |
| | - Directory structure migrated (`pipeline/`, `data/`, `models/`, `scratch/`) | |
| | - Model files consolidated into `models/fvm_distribution/` and `models/multi_feature/` | |
| | - Pipeline notebooks migrated into `pipeline/` and refactored with season parameterization | |
| | - BE contract gap resolved: `Age` (`age`) and `Role_M` (`role_m`) included in `data/final/24-25/players.csv` | |
| | - Fixed `xlsx_to_pkl.py` bug | |
| | - Pipeline executed and verified end-to-end | |
| | - Comprehensive `README.md` created | |

