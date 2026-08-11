import pandas as pd, numpy as np, json
from numpy.polynomial.polynomial import Polynomial

BASE = '/home/duilio999/Documenti/GitProjects/FantaHelpML'
SEASON = '25-26'
INTER_DIR = f'{BASE}/data/intermediate/{SEASON}'
INPUT_PATH = f'{INTER_DIR}/data_preprocess_merge.xlsx'

df = pd.read_excel(INPUT_PATH)
seasons = sorted(set([col[2:] for col in df.columns if col.startswith('Pg')]), reverse=True)
roles_m = ['Por','Dc','B','Dd','E','M','C','W','T','Pc','A']
matches_filter = [8,25,22,29,27,26,28,24,19,27,29]
fvm_curve_min_games = 22

df_m = []
for role in roles_m:
    if role == 'Dd':
        df_temp = df[df['Role_M'].str.split(';').str[0].isin(['Dd','Ds'])]
    else:
        df_temp = df[df['Role_M'].str.split(';').str[0] == role]
    df_m.append(df_temp)

def expected_performance(x, model): return model(x)
def expected_diff_performance(age, model): return expected_performance(age, model) - expected_performance(age-1, model)

stats_models = []
for i, df_temp in enumerate(df_m):
    players = []
    for _, row in df_temp.iterrows():
        for k in seasons:
            if row['Pg'+k] >= matches_filter[i]:
                players.append({'Age':row['Age']-1-seasons.index(k), 'MfPerformance':row['Mf'+k]})
    new_df = pd.DataFrame(players)
    new_df = new_df[new_df['Age'] <= 34]
    avg = new_df.groupby('Age')['MfPerformance'].mean()
    if len(avg) >= 3:
        p = Polynomial.fit(avg.index, avg.values, deg=2)
        stats_models.append(p)
    else:
        stats_models.append(None)

fvm_models = []
for i, df_temp in enumerate(df_m):
    data = []
    for _, row in df_temp.iterrows():
        for j in range(len(seasons)):
            s = seasons[j]
            if pd.notna(row.get(f'Pg{s}')) and row[f'Pg{s}'] >= fvm_curve_min_games:
                fvm = row.get(f'FVM{s}')
                mf = row.get(f'Mf{s}')
                if pd.notna(fvm) and pd.notna(mf) and fvm>0 and mf>0:
                    data.append({'FVM':fvm,'MfPerformance':mf})
    new_df = pd.DataFrame(data)
    if len(new_df) >=10:
        fvm_models.append(np.poly1d(np.polyfit(new_df['FVM'], new_df['MfPerformance'],1)))
    else:
        fvm_models.append(None)

df['ExpectedMf'] = np.nan
role_norm_map = {'Ds':'Dd'}
age_modifier_weight = 0.2
for idx,row in df.iterrows():
    if pd.isna(row.get('Role_M')) or pd.isna(row.get('Age')): continue
    role = row['Role_M'].split(';')[0]
    role = role_norm_map.get(role, role)
    if role not in roles_m: continue
    i = roles_m.index(role)
    fvm_val = row.get('FVM')
    fvm_perf = float(fvm_models[i](fvm_val)) if fvm_models[i] is not None and pd.notna(fvm_val) else 5.0
    age_delta = float(expected_diff_performance(row['Age'], stats_models[i])) if stats_models[i] is not None else 0.0
    expected_mf = fvm_perf + age_modifier_weight * age_delta
    df.at[idx,'ExpectedMf'] = round(expected_mf,2)

def role_key(row):
    rm=str(row.get('Role_M',''))
    first=rm.split(';')[0]
    return role_norm_map.get(first,first)
df['role_key']=df.apply(role_key,axis=1)

out={}
for role in roles_m:
    vals = df.loc[df['role_key']==role,'ExpectedMf'].dropna()
    if len(vals)==0: continue
    out[role]={'count':int(len(vals)),'mean':float(vals.mean()),'std':float(vals.std()),'p5':float(np.percentile(vals,5)),'p95':float(np.percentile(vals,95))}
print(json.dumps(out,indent=2))
with open(f'{INTER_DIR}/validation_stage1_expectedmf_distribution.json','w') as f:
    json.dump(out,f,indent=2)
