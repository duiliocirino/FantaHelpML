# Stage 00 – data_preprocess_merge Refactoring Plan

## Current state
`pipeline/00_data_preprocess_merge.ipynb` merges current season Quotazioni with 5-year historical stats and carries over Age via an ad-hoc `copyCsvReal.xlsx` snapshot from 22-23. Age, Regularness and Mate are then manually updated.

Issues:
* Hidden dependency on `copyCsvReal.xlsx`
* Age carry-forward is implicit and error prone
* No deterministic validation
* No use of historical Quotazioni files

## Target state
Deterministic, reproducible Stage 0 with DOB-based Age and explicit validation.

## Inputs / Outputs
Inputs:
* `data/raw/{SEASON}/Quotazioni_Fantacalcio_Stagione_*.xlsx`
* `data/raw/historical/Quotazioni_Fantacalcio_Stagione_*.xlsx` for previous season
* `data/historical/Statistiche_Fantacalcio_Stagione_*.xlsx`
* `data/utils/player_dob.csv` with Id,Name,DOB

Output:
* `data/intermediate/{SEASON}/data_preprocess_merge.xlsx`
* `data/intermediate/{SEASON}/validation_stage0.json`

Manual step after run: update Regularness and Mate only.

## Changes

### 0A Standardise IO
* First cell defines SEASON and path helpers
* Remove `copyCsvReal.xlsx` usage
* Load current and previous Quotazioni from `data/raw/`

### 0B Age via DOB
* Compute Age from DOB:
  season_start_year = int(SEASON.split("-")[0]) + 2000
  age = season_start_year - dob.year
  if dob.month > 9 or dob.month == 9 and dob.day > 1: age -=1
* Validation: flag missing DOB

### 0C Merge logic
* Map Quotazioni columns via documented dict
* Merge historical stats by Id
* Keep only players present in current Quotazioni

### 0D Validation
* Row count == Quotazioni rows
* Id unique
* Age computed for >99% players
* Stats merged for returning players
* Write validation report

## Data preparation needed
* Populate `data/utils/player_dob.csv` via Wikidata SPARQL + manual review
* Move historical Quotazioni to `data/raw/historical/`
* Ensure `data/historical/` contains stats for seasons up to SEASON-1. Current season stats are not included during trial.

## Acceptance criteria
* Stage 0 runs with SEASON variable only
* No `copyCsvReal` dependency
* Age is automatic, Regularness/Mate remain manual
* Validation report passes
