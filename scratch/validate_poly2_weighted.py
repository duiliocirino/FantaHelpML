import pandas as pd, numpy as np, json
BASE = '/home/duilio999/Documenti/GitProjects/FantaHelpML'
SEASON = '25-26'
INTER_DIR = f'{BASE}/data/intermediate/{SEASON}'
INPUT_PATH = f'{INTER_DIR}/data_preprocess_merge.xlsx'
df = pd.read_excel(INPUT_PATH)
seasons = sorted(set([col[2:] for col in df.columns if col.startswith('Pg')]), reverse=True)
roles_m = ['Por','Dc','B','Dd','E','M','C','W','T','Pc','A']
fvm_curve_min_games = 22

results = {}
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
                data.append((fvm, mf, pg))
    if len(data) < 10:
        results[role] = {'mae': None}
        continue
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    w = np.clip(np.array([d[2] for d in data]), 0, 38)
    coef = np.polyfit(X, y, 2, w=w)
    model = np.poly1d(coef)
    pred = model(X)
    mae = float(np.mean(np.abs(y-pred)))
    results[role] = {'mae': mae, 'n': len(data)}
print(json.dumps(results, indent=2))
with open(f'{INTER_DIR}/validation_stage1_poly2_weighted_mae.json','w') as f:
    json.dump(results, f, indent=2)
