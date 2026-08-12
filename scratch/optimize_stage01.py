#!/usr/bin/env python
"""
Optimize Stage 01 parameters to hit target ExpectedMf values for key players.

Usage:
    poetry run python scratch/optimize_stage01.py
"""
import sys, json
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial

sys.path.insert(0, '.')
from pipeline.fvm_tuning import (
    load_stage0, has_fvm_for_seasons, build_role_subsets,
    collect_fvm_pairs, fit_fvm_model, fit_age_curve,
    resolve_w_fvm, resolve_transform, transform_fvm,
    ROLES_M, ROLE_NORM_MAP, ROLE_TO_MAJOR, MATCHES_FILTERS,
    FVM_MIN_GAMES, AGE_MAX, SEASON_DECAY, MIN_OBS
)

# Target values from user
TARGETS = {
    # Attackers
    'Martinez L.': 8.15,   # target 8-8.3
    'Malen': 8.15,         # target 8-8.3
    'Hojlund': 7.6,
    'Yildiz': 7.6,
    'Raspadori': 7.0,
    'Nkunku': 7.0,
    # Midfielders (W, M, C sub-roles)
    'Orsolini': 7.7,
    'Calhanoglu': 7.3,
    # Attackers (T sub-role = trequartista = attacker)
    'Paz N.': 7.7,
    'Pulisic': 7.3,
    'Zaniolo': 7.0,
    'Baturina': 7.0,
    # Defenders (E sub-role = ala/extramosso = wing-back = defender)
    'Dimarco': 7.8,
    'Palestra': 6.8,
    'Dumfries': 6.8,
    'Bremer': 6.5,
}

INPUT_PATH = 'data/intermediate/25-26/data_preprocess_merge.xlsx'


def compute_expected_mf_custom(df, seasons, fvm_coefs, age_models,
                                blend_weights, transform_spec,
                                age_mod, matches_filters):
    """Compute ExpectedMf with given parameters."""
    result = df.copy()
    result['ExpectedMf'] = np.nan

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

            role_transform = resolve_transform(transform_spec, role)
            fvm_val = row.get('FVM')
            coef = fvm_coefs.get(role)
            if coef is not None and pd.notna(fvm_val):
                x = transform_fvm(float(fvm_val), role_transform)
                fvm_perf = float(np.polyval(coef, x))
            else:
                fvm_perf = 5.0

            mf_last = row.get(f'Mf{seasons[0]}')
            pg_last = row.get(f'Pg{seasons[0]}')
            age_model = age_models.get(role)
            mf_thresh = matches_filters.get(role, 16)

            if (pd.notna(mf_last) and pd.notna(pg_last)
                    and mf_last != 0 and pg_last >= mf_thresh
                    and age_model is not None):
                a_delta = float(age_model(age) - age_model(age - 1))
                hist_perf = float(mf_last) + age_mod * a_delta
            else:
                hist_perf = float(age_model(age)) if age_model is not None else fvm_perf
                if hist_perf == 5.0:
                    hist_perf = fvm_perf

            w = resolve_w_fvm(blend_weights, role)
            expected = w * fvm_perf + (1 - w) * hist_perf
            result.at[idx, 'ExpectedMf'] = round(expected, 2)
        except Exception:
            continue
    return result


def score_config(df, seasons, blend_weights, transform_spec, age_mod,
                 fvm_coefs, age_models, matches_filters):
    """Score how well a config hits the targets. Lower = better."""
    df_out = compute_expected_mf_custom(
        df, seasons, fvm_coefs, age_models,
        blend_weights, transform_spec, age_mod, matches_filters
    )
    total_error = 0
    details = {}
    for name, target in TARGETS.items():
        row = df_out[df_out['Name'] == name]
        if not row.empty:
            actual = row['ExpectedMf'].values[0]
            error = (actual - target) ** 2
            total_error += error
            details[name] = actual
        else:
            total_error += 10  # penalty for missing player
    return total_error / len(TARGETS), details


def main():
    # Load data
    df, seasons = load_stage0(INPUT_PATH)
    fvm_seasons = has_fvm_for_seasons(df, seasons)
    role_subsets = build_role_subsets(df)

    # Fit age curves (fixed across all configs)
    age_models = {}
    for role in ROLES_M:
        age_models[role] = fit_age_curve(
            role_subsets[role], seasons, MATCHES_FILTERS.get(role, 11)
        )

    # Print current player data for debugging
    print("=== Target players data ===")
    for name in TARGETS:
        row = df[df['Name'] == name]
        if not row.empty:
            r = row.iloc[0]
            role = r['Role_M'].split(';')[0]
            fvm = r.get('FVM', 'N/A')
            mf_last = r.get(f'Mf{seasons[0]}', 'N/A')
            pg_last = r.get(f'Pg{seasons[0]}', 'N/A')
            age = r.get('Age', 'N/A')
            print(f"  {name:<15s} Role={role:<4s} FVM={fvm:<8} Age={age:<4} "
                  f"Mf_last={mf_last:<6} Pg_last={pg_last:<6}")
    print()

    # Search strategy: fixed transform, grid blend weights + age_mod
    transform_spec = {'P': 'sqrt', 'D': 'sqrt', 'C': 'linear', 'A': 'linear'}

    # Pre-fit FVM models for this transform
    fvm_coefs = {}
    for role in ROLES_M:
        role_transform = resolve_transform(transform_spec, role)
        data = collect_fvm_pairs(role_subsets[role], fvm_seasons)
        fvm_coefs[role] = fit_fvm_model(data, 'linear', None, role_transform)

    # Reload config to get updated ROLE_TO_MAJOR (E->D)
    from pipeline.fvm_tuning import load_config, _build_aliases
    fresh_cfg = load_config()
    _build_aliases(fresh_cfg)

    # Pre-compute FVM_pred and hist_perf for target players only (speed up)
    target_names = list(TARGETS.keys())
    target_data = {}
    for name in target_names:
        row = df[df['Name'] == name].iloc[0]
        role = row['Role_M'].split(';')[0]
        role = ROLE_NORM_MAP.get(role, role)
        role_transform = resolve_transform(transform_spec, role)
        fvm_val = row.get('FVM')
        coef = fvm_coefs.get(role)
        if coef is not None and pd.notna(fvm_val):
            x = transform_fvm(float(fvm_val), role_transform)
            fvm_perf = float(np.polyval(coef, x))
        else:
            fvm_perf = 5.0
        mf_last = row.get(f'Mf{seasons[0]}')
        pg_last = row.get(f'Pg{seasons[0]}')
        age_model = age_models.get(role)
        if (pd.notna(mf_last) and pd.notna(pg_last)
                and mf_last != 0 and pg_last >= MATCHES_FILTERS.get(role, 16)
                and age_model is not None):
            a_delta = float(age_model(row['Age']) - age_model(row['Age'] - 1))
            hist_base = float(mf_last)
            hist_delta = a_delta
        else:
            hist_base = float(age_model(row['Age'])) if age_model is not None else fvm_perf
            hist_delta = 0
        major = ROLE_TO_MAJOR.get(role, 'C')
        target_data[name] = {
            'fvm_perf': fvm_perf, 'hist_base': hist_base,
            'hist_delta': hist_delta, 'role': major
        }
        print(f"  {name:<15s} FVM_pred={fvm_perf:.2f} hist={hist_base:.2f} delta={hist_delta:.2f} major={major}")

    best_score = float('inf')
    best_cfg = None
    best_details = None

    # Grid search: blend weights + age_mod + fvm_scale (per-role multiplier on FVM_pred)
    # fvm_scale > 1 amplifies FVM predictions to counteract conservative regression
    for age_mod in [x/10 for x in range(0, 5)]:
        for w_d in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for w_c in [0.4, 0.5, 0.6, 0.7, 0.8]:
                for w_a in [0.6, 0.7, 0.8, 0.9, 1.0]:
                    for scale_c in [1.00, 1.05, 1.10, 1.15, 1.20]:
                        for scale_a in [1.00, 1.05, 1.10, 1.15, 1.20]:
                            for scale_d in [1.00, 1.05, 1.10, 1.15]:
                                blend = {'P': 0.8, 'D': w_d, 'C': w_c, 'A': w_a}
                                scales = {'P': 1.0, 'D': scale_d, 'C': scale_c, 'A': scale_a}
                                # Fast score on targets only
                                total_error = 0
                                details = {}
                                for name, target in TARGETS.items():
                                    td = target_data[name]
                                    w = blend[td['role']]
                                    sc = scales[td['role']]
                                    fvm_scaled = td['fvm_perf'] * sc
                                    hist_perf = td['hist_base'] + age_mod * td['hist_delta']
                                    expected = w * fvm_scaled + (1 - w) * hist_perf
                                    details[name] = round(expected, 2)
                                    total_error += (expected - target) ** 2
                                score = total_error / len(TARGETS)
                                if score < best_score:
                                    best_score = score
                                    best_cfg = {
                                        'blend': blend,
                                        'transform': transform_spec,
                                        'age_mod': age_mod,
                                        'fvm_scale': scales,
                                    }
                                    best_details = details

    print(f"=== Best config (RMSE={best_score**0.5:.3f}) ===")
    print(f"  Blend weights: {best_cfg['blend']}")
    print(f"  Transform: {best_cfg['transform']}")
    print(f"  Age modifier: {best_cfg['age_mod']}")
    print()
    print("=== Player predictions vs targets ===")
    for name, target in TARGETS.items():
        actual = best_details.get(name, 'MISSING')
        diff = actual - target if isinstance(actual, (int, float)) else 'N/A'
        marker = ' ✓' if isinstance(diff, (int, float)) and abs(diff) < 0.2 else ''
        print(f"  {name:<15s} actual={actual:<6} target={target:.1f} diff={diff}{marker}")

    # Save to file for reference
    with open('data/intermediate/25-26/optimization_result.json', 'w') as f:
        json.dump({
            'best_config': best_cfg,
            'rmse': best_score**0.5,
            'player_predictions': {k: round(v, 2) for k, v in best_details.items()},
        }, f, indent=2)
    print(f"\nSaved to data/intermediate/25-26/optimization_result.json")


if __name__ == '__main__':
    main()
