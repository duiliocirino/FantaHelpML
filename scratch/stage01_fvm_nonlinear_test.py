import pandas as pd, numpy as np, json
from numpy.polynomial.polynomial import Polynomial

BASE = '/home/duilio999/Documenti/GitProjects/FantaHelpML'
SEASON = '25-26'
INTER_DIR = f'{BASE}/data/intermediate/{SEASON}'
INPUT_PATH = f'{INTER_DIR}/data_preprocess_merge.xlsx'

df = pd.read_excel(INPUT_PATH)
seasons = sorted(set([col[2:] for col in df.columns if col.startswith('Pg')]), reverse=True)
roles_m = ['Por','Dc','B','Dd','E','M','C','W','T','Pc','A']
fvm_curve_min_games = 22

def build_fvm_models(degree):
    models = []
    for role in roles_m:
        if role == 'Dd':
            sub = df[df['Role_M'].str.split(';').str[0].isin(['Dd','Ds'])]
        else:
            sub = df[df['Role_M'].str.split(';').str[0] == role]
        data = []
        for _, row in sub.iterrows():
            for s in seasons:
                fvm = row.get(f'FVM{s}')
                mf = row.get(f'Mf{s}')
                pg = row.get(f'Pg{s}')
                if pd.notna(fvm) and pd.notna(mf) and pd.notna(pg) and pg >= fvm_curve_min_games and fvm>0 and mf>0:
                    data.append((fvm, mf))
        if len(data) < 10:
            models.append(None)
            continue
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        if degree == 1:
            coef = np.polyfit(X, y, 1)
            model = np.poly1d(coef)
        else:
            # polynomial fit with numpy polyfit
            coef = np.polyfit(X, y, degree)
            model = np.poly1d(coef)
        models.append(model)
    return models

def evaluate(models):
    results = {}
    for i, role in enumerate(roles_m):
        model = models[i]
        if model is None:
            results[role] = {'mae': None, 'rmse': None}
            continue
        # collect data again for evaluation
        if role == 'Dd':
            sub = df[df['Role_M'].str.split(';').str[0].isin(['Dd','Ds'])]
        else:
            sub = df[df['Role_M'].str.split(';').str[0] == role]
        xs, ys = [], []
        for _, row in sub.iterrows():
            for s in seasons:
                fvm = row.get(f'FVM{s}')
                mf = row.get(f'Mf{s}')
                pg = row.get(f'Pg{s}')
                if pd.notna(fvm) and pd.notna(mf) and pd.notna(pg) and pg >= fvm_curve_min_games and fvm>0 and mf>0:
                    xs.append(fvm); ys.append(mf)
        xs = np.array(xs); ys = np.array(ys)
        pred = model(xs)
        mae = float(np.mean(np.abs(ys-pred)))
        rmse = float(np.sqrt(np.mean((ys-pred)**2)))
        results[role] = {'mae': mae, 'rmse': rmse, 'n': int(len(xs))}
    return results

linear_models = build_fvm_models(1)
poly2_models = build_fvm_models(2)

linear_res = evaluate(linear_models)
poly2_res = evaluate(poly2_models)

out = {
    'linear': linear_res,
    'poly2': poly2_res
}
with open(f'{INTER_DIR}/validation_stage1_fvm_nonlinear.json','w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
