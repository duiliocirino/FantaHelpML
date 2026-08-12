#!/usr/bin/env python
"""
FVM Model Tuning Framework
===========================
Repeatable preliminary step for Stage 01.

Trains FVM→Mf models under multiple configurations, validates on a hold-out
season (most recent), selects the best config, and retrains on all data.

Usage:
    poetry run python pipeline/fvm_tuning.py          # standalone
    import pipeline.fvm_tuning as fvm_tuning          # as module

Output:
    data/intermediate/{SEASON}/fvm_tuning_report.json
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLES_M = ['Por', 'Dc', 'B', 'Dd', 'Ds', 'E', 'M', 'C', 'W', 'T', 'Pc', 'A']
ROLE_NORM_MAP = {'Ds': 'Dd'}  # merge Ds into Dd for modelling

# Candidate configurations to evaluate
# Each config: dict with keys method, fvm_cap, w_fvm, age_mod
CANDIDATE_CONFIGS = {
    'linear_baseline':      {'method': 'linear',  'fvm_cap': None,  'w_fvm': 0.8, 'age_mod': 0.2},
    'linear_w0.9':          {'method': 'linear',  'fvm_cap': None,  'w_fvm': 0.9, 'age_mod': 0.2},
    'linear_w1.0':          {'method': 'linear',  'fvm_cap': None,  'w_fvm': 1.0, 'age_mod': 0.0},
    'linear_cap200':         {'method': 'linear',  'fvm_cap': 200,   'w_fvm': 0.8, 'age_mod': 0.2},
    'linear_cap200_w0.9':    {'method': 'linear',  'fvm_cap': 200,   'w_fvm': 0.9, 'age_mod': 0.2},
    'theilsen':              {'method': 'theilsen','fvm_cap': None,  'w_fvm': 0.8, 'age_mod': 0.2},
    'theilsen_cap200':       {'method': 'theilsen','fvm_cap': 200,   'w_fvm': 0.8, 'age_mod': 0.2},
}

# Fitting defaults
FVM_MIN_GAMES = 24       # minimum Pg for an observation to enter FVM training
AGE_MIN_GAMES = 16       # minimum Pg for a player to count as "has recent data"
SEASON_DECAY = 0.3       # exponential recency decay for age curve & FVM weights
AGE_MAX = 34             # cap age for polynomial fitting
MIN_OBS = 10             # minimum observations to fit a model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_stage0(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load Stage 0 output and extract available seasons."""
    df = pd.read_excel(path)
    seasons = sorted(
        set(col[2:] for col in df.columns if col.startswith("Pg")),
        reverse=True,
    )
    return df, seasons


def has_fvm_for_seasons(df: pd.DataFrame, seasons: list[str]) -> list[str]:
    """Return seasons that have non-zero FVM columns."""
    available = []
    for s in seasons:
        col = f'FVM{s}'
        if col in df.columns and df[col].gt(0).any():
            available.append(s)
    return available


def build_role_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split dataframe by Role_M, merging Ds into Dd."""
    subsets = {}
    for role in ROLES_M:
        if role == 'Dd':
            mask = df['Role_M'].str.split(';').str[0].isin(['Dd', 'Ds'])
        else:
            mask = df['Role_M'].str.split(';').str[0] == role
        subsets[role] = df[mask]
    return subsets


def collect_fvm_pairs(df_role: pd.DataFrame, seasons: list[str],
                      min_games: int = FVM_MIN_GAMES) -> pd.DataFrame:
    """Collect (FVM, Mf, Pg) pairs from a role subset across given seasons."""
    rows = []
    for _, row in df_role.iterrows():
        for j, s in enumerate(seasons):
            fvm_col, mf_col, pg_col = f'FVM{s}', f'Mf{s}', f'Pg{s}'
            if pg_col not in row.index:
                continue
            try:
                pg_val = row[pg_col]
                if pd.notna(pg_val) and pg_val >= min_games:
                    fvm_val = row.get(fvm_col)
                    mf_val = row.get(mf_col)
                    if (pd.notna(fvm_val) and pd.notna(mf_val)
                            and fvm_val > 0 and mf_val > 0):
                        rows.append({
                            'FVM': fvm_val,
                            'Mf': mf_val,
                            'Pg': pg_val,
                            'season_idx': j,
                        })
            except Exception:
                continue
    return pd.DataFrame(rows)


def collect_age_points(df_role: pd.DataFrame, seasons: list[str],
                       matches_filter: int,
                       decay: float = SEASON_DECAY) -> pd.DataFrame:
    """Collect (Age, Mf, weight) points for age curve fitting."""
    rows = []
    for _, row in df_role.iterrows():
        for k, s in enumerate(seasons):
            pg_col, mf_col = f'Pg{s}', f'Mf{s}'
            if pg_col not in row.index:
                continue
            try:
                if row[pg_col] >= matches_filter:
                    weight = np.exp(-decay * k)
                    rows.append({
                        'Age': row['Age'] - 1 - k,
                        'Mf': row[mf_col],
                        'weight': weight,
                    })
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted linear regression. Returns [slope, intercept]."""
    coef = np.polyfit(x, y, 1, w=w)
    return coef  # poly1d format: [slope, intercept]


def fit_theilsen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Theil-Sen estimator (robust to outliers). Returns [slope, intercept]."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return np.polyfit(x, y, 1)
    med_slope = np.median(slopes)
    med_intercept = np.median(y - med_slope * x)
    return np.array([med_slope, med_intercept])


def fit_fvm_model(data: pd.DataFrame, method: str,
                  fvm_cap: int | None) -> np.ndarray | None:
    """
    Fit a FVM→Mf model.

    - fvm_cap: if set, caps FVM values *during training only* (prevents
      extreme values from flattening the slope).  Predictions use the
      actual FVM (uncapped).
    - method: 'linear' (weighted OLS) or 'theilsen' (robust median slope).
    """
    if len(data) < MIN_OBS:
        return None
    x = data['FVM'].values.copy()
    y = data['MfPerformance'].values if 'MfPerformance' in data.columns else data['Mf'].values
    w = np.clip(data['Pg'].values, 0, 38)

    x_fit = x.copy()
    if fvm_cap is not None:
        x_fit = np.minimum(x_fit, fvm_cap)

    if method == 'theilsen':
        return fit_theilsen(x_fit, y)
    else:
        return fit_linear(x_fit, y, w)


def predict_fvm(coef: np.ndarray, fvm_val: float) -> float:
    """Predict Mf from FVM using fitted linear coefficients."""
    return float(np.polyval(coef, fvm_val))


# ---------------------------------------------------------------------------
# Age curve (quadratic polynomial)
# ---------------------------------------------------------------------------
def fit_age_curve(df_role: pd.DataFrame, seasons: list[str],
                  matches_filter: int):
    """Return fitted Polynomial or None."""
    from numpy.polynomial.polynomial import Polynomial
    pts = collect_age_points(df_role, seasons, matches_filter)
    if len(pts) == 0:
        return None
    pts = pts[pts['Age'] <= AGE_MAX]
    if len(pts) < 3:
        return None
    avg = pts.groupby('Age').apply(
        lambda g: np.average(g['Mf'], weights=g['weight'])
    )
    if len(avg) < 3:
        return None
    return Polynomial.fit(avg.index, avg.values, deg=2)


def age_delta(age: float, model) -> float:
    """Expected performance change from age-1 to age."""
    if model is None:
        return 0.0
    return float(model(age) - model(age - 1))


def age_prediction(age: float, model) -> float:
    """Expected performance at a given age from the curve."""
    if model is None:
        return 5.0
    return float(model(age))


# ---------------------------------------------------------------------------
# ExpectedMf computation
# ---------------------------------------------------------------------------
def compute_expected_mf(df: pd.DataFrame, fvm_coefs: dict[str, np.ndarray | None],
                        age_models: dict[str, any],
                        seasons: list[str],
                        w_fvm: float, age_mod: float,
                        matches_filters: dict[str, int]) -> pd.DataFrame:
    """
    Compute ExpectedMf for every player in df.

    Parameters
    ----------
    fvm_coefs : {role: coef_array | None}
    age_models : {role: Polynomial | None}
    """
    result = df.copy()
    result['ExpectedMf'] = np.nan
    result['FVM_pred'] = np.nan
    result['Hist_perf'] = np.nan

    for idx, row in result.iterrows():
        try:
            role_raw = row.get('Role_M')
            age = row.get('Age')
            if pd.isna(role_raw) or pd.isna(age):
                continue
            role = role_raw.split(';')[0]
            role = ROLE_NORM_MAP.get(role, role)
            if role not in ROLES_M:
                continue

            # FVM prediction
            fvm_val = row.get('FVM')
            coef = fvm_coefs.get(role)
            if coef is not None and pd.notna(fvm_val):
                fvm_perf = predict_fvm(coef, float(fvm_val))
            else:
                fvm_perf = 5.0

            # Historical performance
            mf_last = row.get(f'Mf{seasons[0]}')
            pg_last = row.get(f'Pg{seasons[0]}')
            age_model = age_models.get(role)
            mf_thresh = matches_filters.get(role, AGE_MIN_GAMES)

            if (pd.notna(mf_last) and pd.notna(pg_last)
                    and mf_last != 0 and pg_last >= mf_thresh
                    and age_model is not None):
                a_delta = age_delta(float(age), age_model)
                hist_perf = float(mf_last) + age_mod * a_delta
            else:
                hist_perf = age_prediction(float(age), age_model)
                if pd.isna(hist_perf) or hist_perf == 5.0:
                    hist_perf = fvm_perf

            expected = w_fvm * fvm_perf + (1 - w_fvm) * hist_perf
            result.at[idx, 'ExpectedMf'] = round(expected, 2)
            result.at[idx, 'FVM_pred'] = round(fvm_perf, 2)
            result.at[idx, 'Hist_perf'] = round(hist_perf, 2)
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Hold-out validation
# ---------------------------------------------------------------------------
def validate_holdout(df: pd.DataFrame, seasons: list[str],
                     fvm_coefs: dict[str, np.ndarray | None],
                     age_models: dict[str, any],
                     cfg: dict, matches_filters: dict[str, int]) -> dict:
    """
    Validate on the most recent season.

    For each player with valid FVM and Mf in the hold-out season, compare
    predicted Mf (from FVM model trained on earlier seasons) vs actual Mf.

    Returns per-role MAE, RMSE, and overall metrics.
    """
    holdout = seasons[0]  # most recent
    fvm_col = f'FVM{holdout}'
    mf_col = f'Mf{holdout}'
    pg_col = f'Pg{holdout}'

    per_role = {}
    all_errors = []

    for role in ROLES_M:
        if role == 'Dd':
            mask = df['Role_M'].str.split(';').str[0].isin(['Dd', 'Ds'])
        else:
            mask = df['Role_M'].str.split(';').str[0] == role
        role_df = df[mask]

        preds, actuals = [], []
        for _, row in role_df.iterrows():
            fvm_val = row.get(fvm_col)
            mf_val = row.get(mf_col)
            pg_val = row.get(pg_col)
            if (pd.notna(fvm_val) and pd.notna(mf_val) and pd.notna(pg_val)
                    and fvm_val > 0 and mf_val > 0
                    and pg_val >= FVM_MIN_GAMES):
                coef = fvm_coefs.get(role)
                if coef is not None:
                    pred = predict_fvm(coef, float(fvm_val))
                    preds.append(pred)
                    actuals.append(float(mf_val))

        if len(preds) >= 5:
            preds, actuals = np.array(preds), np.array(actuals)
            mae = float(np.mean(np.abs(actuals - preds)))
            rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
            corr = float(np.corrcoef(preds, actuals)[0, 1])
            per_role[role] = {'n': len(preds), 'mae': mae, 'rmse': rmse, 'corr': corr}
            all_errors.extend(actuals - preds)

    overall_mae = float(np.mean(np.abs(all_errors))) if all_errors else None
    overall_rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2))) if all_errors else None
    total_n = sum(r['n'] for r in per_role.values())

    return {
        'per_role': per_role,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'total_n': total_n,
    }


# ---------------------------------------------------------------------------
# Full tuning pipeline
# ---------------------------------------------------------------------------
MATCHES_FILTERS = {
    'Por': 8, 'Dc': 25, 'B': 22, 'Dd': 29, 'Ds': 29, 'E': 27,
    'M': 26, 'C': 28, 'W': 24, 'T': 19, 'Pc': 27, 'A': 29,
}


def run_tuning(input_path: str, output_dir: str, verbose: bool = True):
    """
    Full tuning pipeline.

    1. Load data, identify seasons with FVM
    2. For each candidate config:
       a. Train FVM models on all-but-last FVM season
       b. Validate on last FVM season (hold-out)
       c. Record MAE/RMSE
    3. Select best config (lowest overall MAE)
    4. Retrain best config on ALL FVM seasons
    5. Compute final ExpectedMf with best config
    6. Save report + coefficients

    Returns path to report JSON.
    """
    df, seasons = load_stage0(input_path)
    fvm_seasons = has_fvm_for_seasons(df, seasons)

    if verbose:
        print(f"Seasons available: {seasons}")
        print(f"Seasons with FVM:  {fvm_seasons}")
        print(f"Candidate configs: {list(CANDIDATE_CONFIGS.keys())}")
        print()

    if len(fvm_seasons) < 2:
        if verbose:
            print("WARNING: Fewer than 2 FVM seasons available. "
                  "Skipping hold-out validation, using all data.")
        train_seasons = fvm_seasons
        holdout_seasons = []
    else:
        train_seasons = fvm_seasons[1:]   # all except most recent
        holdout_seasons = [fvm_seasons[0]]  # most recent

    role_subsets = build_role_subsets(df)
    results = {}

    # --- Phase 1: evaluate each config on hold-out ---
    for name, cfg in CANDIDATE_CONFIGS.items():
        # Train on all-but-last FVM season
        coefs = {}
        for role in ROLES_M:
            data = collect_fvm_pairs(role_subsets[role], train_seasons)
            coefs[role] = fit_fvm_model(data, cfg['method'], cfg['fvm_cap'])

        # Validate on hold-out
        if holdout_seasons:
            val = validate_holdout(
                df, fvm_seasons, coefs, {}, cfg, MATCHES_FILTERS
            )
        else:
            val = {'per_role': {}, 'overall_mae': None, 'overall_rmse': None, 'total_n': 0}

        results[name] = {
            'config': cfg,
            'validation': val,
            'coefs_train': {r: c.tolist() if c is not None else None for r, c in coefs.items()},
        }
        if verbose:
            status = (f"MAE={val['overall_mae']:.4f} RMSE={val['overall_rmse']:.4f} "
                      f"n={val['total_n']}" if val['overall_mae'] else "N/A (no holdout)")
            print(f"  {name:<22s}: {status}")

    if verbose:
        print()

    # --- Phase 2: select best config ---
    # Primary: lowest MAE; secondary: highest total_n (more data = more reliable)
    scored = []
    for name, res in results.items():
        mae = res['validation']['overall_mae']
        n = res['validation']['total_n']
        if mae is not None:
            scored.append((mae, -n, name))
        else:
            scored.append((float('inf'), 0, name))
    scored.sort()
    best_name = scored[0][2]

    if verbose:
        print(f"Best config: {best_name} "
              f"(MAE={results[best_name]['validation']['overall_mae']})")
        print()

    # --- Phase 3: retrain best config on ALL FVM seasons ---
    best_cfg = CANDIDATE_CONFIGS[best_name]
    final_coefs = {}
    for role in ROLES_M:
        data = collect_fvm_pairs(role_subsets[role], fvm_seasons)
        final_coefs[role] = fit_fvm_model(data, best_cfg['method'], best_cfg['fvm_cap'])

    # --- Phase 4: fit age curves on ALL seasons ---
    age_models = {}
    for role in ROLES_M:
        age_models[role] = fit_age_curve(
            role_subsets[role], seasons, MATCHES_FILTERS.get(role, 11)
        )

    # --- Phase 5: compute ExpectedMf ---
    result_df = compute_expected_mf(
        df, final_coefs, age_models, seasons,
        w_fvm=best_cfg['w_fvm'], age_mod=best_cfg['age_mod'],
        matches_filters=MATCHES_FILTERS,
    )

    # --- Phase 6: build report ---
    # ExpectedMf distribution stats
    emf_stats = {}
    for role in ['P', 'D', 'C', 'A']:
        vals = result_df.loc[result_df['Role'] == role, 'ExpectedMf'].dropna()
        if len(vals):
            emf_stats[role] = {
                'count': int(len(vals)),
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'p5': round(float(vals.quantile(0.05)), 4),
                'p95': round(float(vals.quantile(0.95)), 4),
            }

    # Final coefs
    final_coef_report = {}
    for role in ROLES_M:
        c = final_coefs[role]
        if c is not None:
            final_coef_report[role] = {'slope': round(float(c[0]), 6),
                                        'intercept': round(float(c[1]), 4)}
        else:
            final_coef_report[role] = None

    report = {
        'season': seasons[0] if seasons else 'unknown',
        'fvm_seasons': fvm_seasons,
        'train_seasons': train_seasons,
        'holdout_seasons': holdout_seasons,
        'best_config_name': best_name,
        'best_config': best_cfg,
        'config_comparison': {
            name: {
                'config': res['config'],
                'holdout_mae': res['validation']['overall_mae'],
                'holdout_rmse': res['validation']['overall_rmse'],
                'holdout_n': res['validation']['total_n'],
                'per_role': res['validation']['per_role'],
            }
            for name, res in results.items()
        },
        'final_coefs': final_coef_report,
        'expectedmf_stats': emf_stats,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'fvm_tuning_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"Report saved: {report_path}")
        print(f"ExpectedMf stats:")
        for role, st in emf_stats.items():
            print(f"  {role}: mean={st['mean']:.2f} std={st['std']:.3f} "
                  f"p5={st['p5']:.2f} p95={st['p95']:.2f}")

    # Also save the result dataframe for the notebook to pick up
    emf_path = os.path.join(output_dir, 'expected_mf_tuned.csv')
    result_df.to_csv(emf_path, index=False)
    if verbose:
        print(f"Tuned ExpectedMf saved: {emf_path}")

    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FVM Model Tuning Framework')
    parser.add_argument('--input', default=None,
                        help='Path to Stage 0 output (data_preprocess_merge.xlsx)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for report output')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    run_tuning(args.input, args.output_dir, verbose=not args.quiet)
