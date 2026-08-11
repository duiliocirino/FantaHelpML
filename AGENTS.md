# FantaHelpML Agent Guidelines

## Repository purpose
Machine learning pipeline for Fantacalcio player evaluation, rating, and expected auction price estimation. Output is BE-ready CSV consumed by FantaHelpBE.

## Core structure
```
pipeline/
  00_data_preprocess_merge.ipynb  # Stage 0
  01_ratings.ipynb                # Stage 1
  02_fvm_to_distribution.ipynb    # Stage 2A
  03_regressors.ipynb             # Stage 2B - active final output
  04_expected_price.ipynb         # predecessor
data/
  raw/{SEASON}/          # unmodified source
  intermediate/{SEASON}/ # stage outputs
  final/{SEASON}/        # BE-ready players.csv
  historical/            # 5 seasons stats
models/
  fvm_distribution/{SEASON}/ # joblib mean/std per role
  multi_feature/           # lasso/ridge experiments
explorations/            # validation / post-hoc
scratch/                 # temp work, safe landing for remote files
```

## Season handling
* First cell of every pipeline notebook defines `SEASON = "xx-yy"` and derives paths:
  `RAW_DIR = f"data/raw/{SEASON}"`, `INTER_DIR = f"data/intermediate/{SEASON}"`, etc.
* Never hardcode `24-25`. Use the variable everywhere.
* Final output filename is season aware, e.g. `data/final/{SEASON}/players.csv`.

## Input / Output contracts
Stage 0: reads `data/raw/{SEASON}/Quotazioni_*.xlsx`, `copyCsvReal.xlsx`, `data/historical/*`. Writes `data/intermediate/{SEASON}/data_preprocess_merge.xlsx`. Manual update of Age, Regularness, Mate required after this stage.

Stage 1: reads `data/intermediate/{SEASON}/data_preprocess_merge.xlsx`. Writes `data/intermediate/{SEASON}/output_rp.csv` with columns Id,Role,Role_M,Name,Squad,Price,Age,MyRating,Mate,Regularness,FVM,ExpectedMf + historical stats.

Stage 2A: reads `data/intermediate/{SEASON}/output_rp.csv` + all `data/raw/{SEASON}/Rose_*.xlsx`. Writes `models/fvm_distribution/{SEASON}/model_mean/std_[P,D,C,A].joblib`.

Stage 2B: reads `output_rp.csv` + models. Writes `data/final/{SEASON}/players.csv` with BE columns:
`id,role,role_m,name,squad,price,age,myrating,mate,regularness,fvm,expmf,expprice,expstd`

## BE contract must hold
CsvHelper matches case-insensitively. Keep column names exactly as above. `role_m` is semicolon-separated sub-positions. `mate` is player name for now. `age` and `role_m` must be present.

## Working rules
* Read before edit. Never edit a file you haven't read in this session.
* Change minimally, match existing style.
* No commits on your own. Never push.
* Use `scratch/` for prototypes, fetched data, repro tests.
* Manual intervention points are by design: after Stage 0 update Age/Regularness/Mate.
* Keep notebooks deterministic: set SEASON, avoid relative path magic.
* Validation is mandatory per stage. Do not claim a stage works without running a smoke test.
* Run Python via `poetry run python` and notebooks via `poetry run jupyter` to ensure dependencies.
* Prefer repo tools `read_file`, `edit`, `grep` over inline Python for file edits; use bash only when a dedicated tool cannot do the task.

## Docs as source of truth
Agents working in this repo must read and keep these documents in sync with the actual codebase:
* `docs/improvement-roadmap.md` — overall pipeline improvement plan, progress tracking, next actions
* `docs/stage01_improvement_plan.md` — Stage 01 detailed improvement phases, status table, experiment tracking
* Notebook markdown cells — methodology explanations inside notebooks must match what the code actually does

When applying a change that affects methodology, parameters, or pipeline flow:
1. Read the relevant doc(s) first to understand current state.
2. Update the doc(s) alongside the code change — never let them drift.
3. If a planned phase item is completed or abandoned, mark it in the status table.

If docs and code contradict each other, or if the intended behavior is unclear, ask the user before proceeding — do not assume.

## Validation expectations per stage
* Stage 0: row count matches Quotazioni, Age incremented correctly, no duplicate Id, historical stats merged for players present in previous season.
* Stage 1: ExpectedMf in plausible range, MyRating 1-5, no NaNs in FVM/Price for active players.
* Stage 2A: training data has count>=8 filter, per-role FVM floor applied, models load and predict without error, mean>=1, std>=1.
* Stage 2B: final CSV schema matches BE contract, expprice/expstd >0, no duplicate Id, spot-check top players.

## Coding standards
Clear typing, clarity, separation of concerns, modularity, documentation. Simplicity over over-engineering. After first version, review and reduce complexity. Smoke test changes.

## Integration of uncommitted work
Remote work lands in `scratch/`. Compare, cherry-pick improvements, move data to proper `data/` locations. Do not merge directly into pipeline without validation.

## What not to do
* Do not rename columns in final CSV to match BE internally – keep BE names.
* Do not drop Age/Role_M.
* Do not train on a single league only; prefer multi-league aggregation.
