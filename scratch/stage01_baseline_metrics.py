import os, json, pandas as pd, numpy as np
from pathlib import Path

BASE = '/home/duilio999/Documenti/GitProjects/FantaHelpML'
SEASON = '25-26'
INTER_DIR = f'{BASE}/data/intermediate/{SEASON}'
INPUT_PATH = f'{INTER_DIR}/data_preprocess_merge.xlsx'

df = pd.read_excel(INPUT_PATH)

# Extract seasons from Pg columns
seasons = sorted(set([col[2:] for col in df.columns if col.startswith('Pg')]), reverse=True)
print('Seasons:', seasons)

# Role mapping
roles_m = ['Por','Dc','B','Dd','E','M','C','W','T','Pc','A']
role_norm_map = {'Ds':'Dd'}

# Subset per role
def role_key(row):
    rm = str(row.get('Role_M',''))
    first = rm.split(';')[0]
    return role_norm_map.get(first, first)

df['role_key'] = df.apply(role_key, axis=1)

metrics = {}
for role in roles_m:
    sub = df[df['role_key']==role]
    xs, ys = [], []
    for _, row in sub.iterrows():
        for s in seasons:
            fvm_col = f'FVM{s}'
            mf_col = f'Mf{s}'
            pg_col = f'Pg{s}'
            if fvm_col in row and mf_col in row and pg_col in row:
                fvm = row[fvm_col]
                mf = row[mf_col]
                pg = row[pg_col]
                if pd.notna(fvm) and pd.notna(mf) and pd.notna(pg) and pg >= 22 and fvm>0 and mf>0:
                    xs.append(fvm); ys.append(mf)
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0,1]
        # simple linear fit for RMSE
        coef = np.polyfit(xs, ys, 1)
        pred = np.polyval(coef, xs)
        rmse = np.sqrt(np.mean((ys-pred)**2))
        mae = np.mean(np.abs(ys-pred))
    else:
        corr = None; rmse = None; mae = None
    metrics[role] = {
        'n_pairs': int(len(xs)),
        'corr': float(corr) if corr is not None else None,
        'rmse': float(rmse) if rmse is not None else None,
        'mae': float(mae) if mae is not None else None
    }

out = {
    'season': SEASON,
    'seasons': seasons,
    'metrics_per_role': metrics
}
out_path = f'{INTER_DIR}/validation_stage1_baseline.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print('Saved', out_path)
print(json.dumps(metrics, indent=2))
