# Stage 01 Review — Current State and Improvement Analysis

## What Works Well

- Hybrid blend (FVM + historical) is sound in concept
- Season recency decay (0.3) correctly weights recent seasons higher
- Ds → Dd merge increases sample size meaningfully
- FVM same-season pooling (`FVM_t → Mf_t`) is confirmed correct
- Age as delta modifier (weight 0.2) — correct direction
- Visualizations with role colors, correlation badges — solid
- Notebook structure — clean, well-documented, readable

## Current Results

| Metric                  | Value                                              |
| ----------------------- | -------------------------------------------------- |
| Overall ExpectedMf std  | 0.586 (user wants ~0.4 more spread)                |
| A role std              | 0.276                                              |
| C role std              | 0.252                                              |
| D role std              | 0.204                                              |
| P role std              | 0.181                                              |
| Top player (Dimarco)    | 8.16                                               |
| Max-Min range           | 3.52                                               |

## Root Cause of Conservatism

The FVM → Mf linear regression slope is pulled down by extreme FVM values. FVM ranges 1–450 but Mf only 5–9. Players with FVM=300–450 have Mf of 7–8.5 — valid data points, but they flatten the regression slope. The result: a shallow slope (0.003–0.008) that compresses predictions.

For Pc (main striker): `slope=0.0041` means every 100 FVM only adds 0.41 Mf. For A (winger/forward): `slope=0.0052`.

## Experiment Results — 8 Configurations Tested

| Config        | Overall Std | A Std  | Top 3 Players                                  | Verdict                    |
| ------------- | ----------- | ------ | ---------------------------------------------- | -------------------------- |
| baseline      | 0.586       | 0.276  | Dimarco=8.16, Malen=8.01, Martinez=7.65        | Current                    |
| w_fvm=0.9     | 0.585       | 0.281  | Dimarco=8.32, Malen=8.11                       | Minimal gain               |
| w_fvm=1.0     | 0.586       | 0.294  | Dimarco=8.49, Malen=8.21                       | Better but no history      |
| fvm_cap_200   | 0.595       | 0.396  | Martinez=8.37, Malen=8.32                      | Good spread gain           |
| fvm_cap_150   | 0.617       | 0.528  | Martinez=9.02, Malen=8.83                      | Too aggressive             |
| theilsen      | 0.594       | 0.303  | Dimarco=8.31, Malen=8.09                       | Moderate                   |
| trimmed       | 0.625       | 0.292  | McTominay=8.30                                 | Unstable (negative slopes) |
| cap200 + w0.9 | 0.599       | 0.425  | Martinez=8.49, Malen=8.46, Dimarco=8.32        | **Best balance**           |

> **Note:** Of these 8, 7 are included in `pipeline/fvm_tuning.py`'s `CANDIDATE_CONFIGS` as the repeatable tuning framework. `fvm_cap_150` and `trimmed` were exploratory — too aggressive or unstable to include as production candidates.

## Recommendation

**FVM cap during training (200) + w_fvm=0.9** gives the best balance:

- Caps FVM at 200 during regression fitting (prevents extreme values from flattening slope)
- Predictions use actual FVM (uncapped) — so high-FVM players benefit from the steeper slope
- Increases A role std from 0.276 → 0.425 (+54%)
- Top players: Martinez=8.49, Malen=8.46, Dimarco=8.32 — more differentiated
- Values remain plausible (no 9+ predictions)
- Overall std: 0.599 (modest overall increase, but concentrated where it matters — A and C roles)

### Why This Works

Capping FVM at 200 during training removes the leverage of extreme outliers (FVM 200–450) on the regression slope. The fitted line becomes steeper because it's fitted on the "main body" of the data. During prediction, the steeper line is applied to the actual FVM (315, 350), producing higher predictions for elite players. This is valid because the FVM scale in Fantacalcio is inherently compressed at the top — a player with FVM=350 isn't 3.5x better than FVM=100 in terms of Mf.

## Additional Theoretical Improvements (Not Tested Yet)

1. **ΔFVM feature** — players whose FVM increased significantly last season may be on an upward trajectory
2. **Role-specific w_fvm** — high-correlation roles (A=0.85, T=0.84, Pc=0.83) could use higher FVM weight
3. **Player-specific shrinkage** — for players with 2+ seasons of data, blend their personal mean toward the role mean
4. **Hold-out validation** — train on 22-23 / 23-24, validate on 24-25 actual Mf to pick the best config objectively
