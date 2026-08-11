import pandas as pd, numpy as np, json
BASE = '/home/duilio999/Documenti/GitProjects/FantaHelpML'
SEASON = '25-26'
INTER_DIR = f'{BASE}/data/intermediate/{SEASON}'
INPUT_PATH = f'{INTER_DIR}/data_preprocess_merge.xlsx'
df = pd.read_excel(INPUT_PATH)
seasons = sorted(set([col[2:] for col in df.columns if col.startswith('Pg')]), reverse=True)
roles_m = ['Por','Dc','B','Dd','E','M','C','W','T','Pc','A']
fvm_curve_min_games = 22

def build_models(use_delta):
    models = {}
    for role in roles_m:
        if role == 'Dd':
            sub = df[df['Role_M'].str.split(';').str[0].isin(['Dd','Ds'])]
        else:
            sub = df[df['Role_M'].str.split(';').str[0] == role]
        data = []
        for _, row in sub.iterrows():
            # need at least two seasons for delta
            for i in range(len(seasons)-1):
                s_cur = seasons[i]
                s_prev = seasons[i+1]
                fvm_cur = row.get(f'FVM{s_cur}')
                fvm_prev = row.get(f'FVM{s_prev}')
                mf_cur = row.get(f'Mf{s_cur}')
                pg_cur = row.get(f'Pg{s_cur}')
                if pd.notna(fvm_cur) and pd.notna(mf_cur) and pd.notna(pg_cur) and pg_cur >= fvm_curve_min_games:
                    if use_delta and pd.notna(fvm_prev):
                        delta = fvm_cur - fvm_prev
                        data.append((fvm_cur, delta, mf_cur))
                    elif not use_delta:
                        data.append((fvm_cur, mf_cur))
        if len(data) < 10:
            models[role] = None
            continue
        if use_delta:
            X = np.array([[d[0], d[1]] for d in data])
            y = np.array([d[2] for d in data])
            # simple linear regression with two features
            X_aug = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            models[role] = {'coef': coef}
        else:
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            coef = np.polyfit(X, y, 2)
            models[role] = {'coef': coef, 'deg':2}
    return models

def evaluate(models, use_delta):
    results = {}
    for role in roles_m:
        model = models.get(role)
        if model is None:
            results[role] = {'mae': None}
            continue
        if role == 'Dd':
            sub = df[df['Role_M'].str.split(';').str[0].isin(['Dd','Ds'])]
        else:
            sub = df[df['Role_M'].str.split(';').str[0] == role]
        xs, ys = [], []
        for _, row in sub.iterrows():
            for i in range(len(seasons)-1):
                s_cur = seasons[i]
                s_prev = seasons[i+1]
                fvm_cur = row.get(f'FVM{s_cur}')
                fvm_prev = row.get(f'FVM{s_prev}')
                mf_cur = row.get(f'Mf{s_cur}')
                pg_cur = row.get(f'Pg{s_cur}')
                if pd.notna(fvm_cur) and pd.notna(mf_cur) and pd.notna(pg_cur) and pg_cur >= fvm_curve_min_games:
                    if use_delta and pd.notna(fvm_prev):
                        delta = fvm_cur - fvm_prev
                        xs.append([fvm_cur, delta])
                        ys.append(mf_cur)
                    elif not use_delta:
                        xs.append(fvm_cur)
                        ys.append(mf_cur)
        if not xs:
            results[role] = {'mae': None}
            continue
        if use_delta:
            X = np.array(xs)
            y = np.array(ys)
            X_aug = np.column_stack([np.ones(len(X)), X])
            pred = X_aug @ model['coef']
        else:
            X = np.array(xs)
            y = np.array(ys)
            model_poly = np.poly1d(model['coef'])
            pred = model_poly(X)
        mae = float(np.mean(np.abs(y-pred)))
        results[role] = {'mae': mae, 'n': len(xs)}
    return results

base = evaluate(build_models(False), False)
delta = evaluate(build_models(True), True)

out = {'base': base, 'delta': delta}
with open(f'{INTER_DIR}/validation_stage1_delta_fvm.json','w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
