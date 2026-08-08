# FantaHelpML

Machine learning pipeline for evaluation, rating, and expected auction price estimation of Fantacalcio players.

## Repository Layout

```
FantaHelpML/
├── README.md                       # Pipeline documentation & annual guide
├── docs/
│   └── refactoring-plan.md         # Architecture & refactoring plan
├── pipeline/                       # Sequential pipeline notebooks
│   ├── 00_data_preprocess_merge.ipynb   # Stage 0: Raw data ingestion & historical merge
│   ├── 01_ratings.ipynb                 # Stage 1: Performance curves & player ratings
│   ├── 02_fvm_to_distribution.ipynb     # Stage 2A: Train linear models from real auction data
│   ├── 03_regressors.ipynb              # Stage 2B: Active production price estimation & BE export
│   └── 04_expected_price.ipynb          # Stage 3: Predecessor multi-feature Lasso pipeline
├── data/
│   ├── raw/                        # Season raw inputs (unmodified source files)
│   │   └── 24-25/
│   │       ├── Quotazioni_Fantacalcio_Stagione_2024_25.xlsx
│   │       ├── Rose_fantalega-nicosia.xlsx
│   │       ├── squads.csv
│   │       └── copyCsvReal.xlsx
│   ├── intermediate/               # Intermediate stage outputs per season
│   │   └── 24-25/
│   │       ├── data_preprocess_merge.xlsx  # Stage 0 output
│   │       └── output_rp.csv               # Stage 1 output
│   ├── final/                      # Production BE-ready exports
│   │   └── 24-25/
│   │       └── players.csv         # Importable CSV for backend API
│   ├── historical/                 # Stats from past 5 seasons (2019-2024)
│   └── utils/                      # Utility scripts
│       └── xlsx_to_pkl.py
├── models/                         # Trained model artifacts
│   ├── fvm_distribution/           # LinearRegression models (mean & std per role)
│   │   └── 24-25/
│   └── multi_feature/              # Predecessor Lasso and Ridge models
└── scratch/                        # Archive for legacy data/trial runs
```

---

## Pipeline Stages & Contracts

```
Stage 0: pipeline/00_data_preprocess_merge.ipynb
  Reads:  data/raw/{SEASON}/Quotazioni_*.xlsx
          data/raw/{SEASON}/copyCsvReal.xlsx
          data/historical/*.xlsx
  Writes: data/intermediate/{SEASON}/data_preprocess_merge.xlsx
  Note:   Manual inspection/update of Age, Regularness, Mate can be done after this stage.

Stage 1: pipeline/01_ratings.ipynb
  Reads:  data/intermediate/{SEASON}/data_preprocess_merge.xlsx
  Writes: data/intermediate/{SEASON}/output_rp.csv

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
Using `uv`:
```bash
uv venv .venv
uv pip install --python .venv/bin/python pandas numpy matplotlib joblib scikit-learn openpyxl nbformat
```

### 2. Execution
Run pipeline notebooks sequentially:
```bash
.venv/bin/python -c "
import json

def run_nb(path):
    with open(path) as f: nb = json.load(f)
    gl = {'__name__': '__main__'}
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            code = ''.join([l for l in c['source'] if not l.strip().startswith('%')])
            if code.strip(): exec(code, gl)

run_nb('pipeline/00_data_preprocess_merge.ipynb')
run_nb('pipeline/01_ratings.ipynb')
run_nb('pipeline/02_fvm_to_distribution.ipynb')
run_nb('pipeline/03_regressors.ipynb')
"
```

---

## Annual Season Transition Guide

1. **Update historical stats**: Add previous season stats to `data/historical/`.
2. **Collect new raw data**: Place source files in `data/raw/{new-season}/`:
   - `Quotazioni_Fantacalcio_Stagione_*.xlsx`
   - `Rose_*.xlsx` (real auction data)
   - `squads.csv` (team strength indices)
   - `copyCsvReal.xlsx` (merged dataset from previous season)
3. **Update season variable**: Set `SEASON = "xx-xy"` at top of pipeline notebooks.
4. **Execute pipeline**: Run stages `00` through `03`.
5. **Import into Backend**: Upload `data/final/{new-season}/players.csv` to the backend API (`POST /api/players/import`).