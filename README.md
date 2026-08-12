# FantaHelpML

Machine learning pipeline for evaluation, rating, and expected auction price estimation of Fantacalcio players.

## Repository Layout

```
FantaHelpML/
├── README.md                       # Pipeline documentation
├── docs/                           # Design docs & improvement tracking
│   ├── improvement-roadmap.md
│   ├── stage01_improvement_plan.md
│   └── stage01_analysis.md
├── pipeline/                       # Sequential pipeline notebooks
│   ├── config/
│   │   └── stage01.yaml            # Centralized Stage 1 config
│   ├── fvm_tuning.py               # Automated hold-out tuning framework
│   ├── 00_data_preprocess_merge.ipynb   # Stage 0: Raw data ingestion & historical merge
│   ├── 01_ratings.ipynb                 # Stage 1: Performance curves & player ratings
│   ├── 02_fvm_to_distribution.ipynb     # Stage 2A: Train models from real auction data
│   ├── 03_regressors.ipynb              # Stage 2B: Production price estimation & BE export
│   └── 04_expected_price.ipynb          # Predecessor multi-feature Lasso pipeline
├── data/
│   ├── raw/{SEASON}/               # Season raw inputs (unmodified source files)
│   ├── raw/historical/             # Historical Quotazioni files
│   ├── intermediate/{SEASON}/      # Intermediate stage outputs per season
│   ├── final/{SEASON}/             # Production BE-ready exports (players.csv)
│   ├── historical/                 # Seasonal stats (Pg, Mv, Mf) from past seasons
│   └── utils/
│       └── player_dob.csv          # Player Name, Id, Date of Birth
├── models/                         # Trained model artifacts
│   └── fvm_distribution/{SEASON}/  # FVM distribution models (mean & std per role)
└── scratch/                        # Experimental scripts & trial runs
```

---

## Pipeline Stages & Contracts

```
Stage 0: pipeline/00_data_preprocess_merge.ipynb
  Reads:  data/raw/{SEASON}/Quotazioni_*.xlsx
          data/utils/player_dob.csv
          data/historical/*.xlsx
          data/raw/historical/*.xlsx
  Writes: data/intermediate/{SEASON}/data_preprocess_merge.xlsx
  Note:   Age computed from DOB. Manual update of Regularness/Mate after this stage.

Stage 1: pipeline/01_ratings.ipynb
  Reads:  data/intermediate/{SEASON}/data_preprocess_merge.xlsx
          pipeline/config/stage01.yaml
          pipeline/fvm_tuning.py
  Writes: data/intermediate/{SEASON}/output_rp.csv
  Note:   Automated hold-out tuning, per-role FVM transforms, hybrid ExpectedMf blend.

Stage 2A: pipeline/02_fvm_to_distribution.ipynb
  Reads:  data/intermediate/{SEASON}/output_rp.csv
          data/raw/{SEASON}/Rose_*.xlsx
  Writes: models/fvm_distribution/{SEASON}/model_[mean,std]_[P,D,C,A].joblib

Stage 2B: pipeline/03_regressors.ipynb  (Active Final Production Output)
  Reads:  data/intermediate/{SEASON}/output_rp.csv
          models/fvm_distribution/{SEASON}/model_*.joblib
  Writes: data/final/{SEASON}/players.csv

Stage 3: pipeline/04_expected_price.ipynb  (Predecessor Multi-Feature Model)
  Reads:  data/intermediate/{SEASON}/output_rp.csv
          models/multi_feature/*.pkl
          data/raw/{SEASON}/squads.csv
  Writes: data/final/{SEASON}/players_lasso.csv
```

---

## Final Output CSV Format

The file `data/final/{SEASON}/players.csv` produced by `03_regressors.ipynb` matches the backend `PlayerCreateDto` schema:

| Column Header | Type | Description |
|---|---|---|
| `id` | int | Official Fantacalcio Player ID |
| `role` | string | Role (P, D, C, A) |
| `role_m` | string | Mantra Role (e.g. `Por`, `M;C`, `W;A`) |
| `name` | string | Player Name |
| `squad` | string | Serie A Team |
| `price` | int | Current Quotation (Initial Price) |
| `age` | int / null | Player Age |
| `myrating` | float / null | Model Rating (1 to 10 scale) |
| `mate` | float / null | Teammate performance modifier |
| `regularness` | float / null | Regularity score |
| `fvm` | float | Fantacalcio Market Value Index |
| `expmf` | float / null | Expected Fantavoto |
| `expprice` | int | Expected Auction Price (Mean) |
| `expstd` | int | Expected Auction Price (Standard Deviation) |

---

## Setup & Running Instructions

### 1. Environment Setup
```bash
poetry install
```

### 2. Execution
Run pipeline notebooks sequentially via Jupyter (from repo root):
```bash
poetry run jupyter notebook
```
Or run individual notebooks headlessly:
```bash
poetry run jupyter nbconvert --to notebook --execute pipeline/00_data_preprocess_merge.ipynb
poetry run jupyter nbconvert --to notebook --execute pipeline/01_ratings.ipynb
# ... continue with 02 and 03
```

---

## Annual Season Transition Guide

1. **Update historical stats**: Add previous season stats to `data/historical/`.
2. **Update player DOB**: Add new players to `data/utils/player_dob.csv` (or use `scratch/build_player_dob.py` via Wikidata).
3. **Collect new raw data**: Place source files in `data/raw/{new-season}/`:
   - `Quotazioni_Fantacalcio_Stagione_*.xlsx`
   - `Rose_*.xlsx` (real auction data)
4. **Update season variable**: Set `SEASON = "xx-yy"` in notebook first cells and `pipeline/config/stage01.yaml` if needed.
5. **Execute pipeline**: Run stages `00` through `03`.
6. **Import into Backend**: Upload `data/final/{new-season}/players.csv` to the backend API (`POST /api/players/import`).