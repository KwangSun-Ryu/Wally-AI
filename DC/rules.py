# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

# Data path
input_path = "data/original/train.csv"

# Load data
data = pd.read_csv(input_path)
print(data)

##### Data specific conversion from numerical to categorical #####

def a_year_event(df):
    if df['a_year_event'] == 0:
        return 'Survived'
    elif df['a_year_event'] == 1:
        return 'Death'

def gender(df):
    if df['gender'] == 1:
        return 'Male'
    elif df['gender'] == 2:
        return 'Female'

def age(df):
    if df['age'] <= 29:
        return '<=29'
    elif (df['age'] >= 30) and (df['age'] <= 39):
        return '30~39'
    elif (df['age'] >= 40) and (df['age'] <= 49):
        return '40~49'
    elif (df['age'] >= 50) and (df['age'] <= 59):
        return '50~59'
    elif (df['age'] >= 60) and (df['age'] <= 69):
        return '60~69'
    elif (df['age'] >= 70) and (df['age'] <= 79):
        return '70~79'
    elif df['age'] >= 80:
        return '>=80'

def smoker(df):
    if df['Smoker'] == 1:
        return 'Non-smoker'
    elif df['Smoker'] > 1:
        return 'Smoker'

def packyear(df):
    if df['PackYear'] == 0:
        return 'Never-smoked'
    elif df['PackYear'] > 0:
        return 'Ever-smoked'
    
def quantile_discrete(df, float_columns):
    discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy='uniform')
    return discretizer.fit_transform(df[float_columns].values)
    
data['a_year_event'] = data.apply(a_year_event, axis=1)
data['gender'] = data.apply(gender, axis=1)
data['age'] = data.apply(age, axis=1)
data['Smoker'] = data.apply(smoker, axis=1)
data['PackYear'] = data.apply(packyear, axis=1)

float_columns = ['HEIGHT', 'WEIGHT', 'FVC', 'FEV1', 'DLCO', 'DLCO_PERCENT']
data[float_columns] = quantile_discrete(data, float_columns)

##################################################################

##################### Cramer's V Correlation #####################

def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1
    return (stat / (obs * mini))

rows = []
for var1 in data:
    col = []
    for var2 in data:
        cramers = cramers_V(data[var1], data[var2])
        col.append(round(cramers, 2))
    rows.append(col)

cramers_results = np.array(rows)
corr_df = pd.DataFrame(cramers_results, columns=data.columns, index=data.columns)
print(corr_df)
print(pd.crosstab(data["Smoker"], data["PackYear"]))

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(corr_df, 10))

##################################################################

######################## Correlation Plot ########################

new_index = {
    "a_year_event": "A-year-event [A]",
    "gender": "Gender [B]",
    "age": "Age [C]",
    "HEIGHT": "Height [D]",
    "WEIGHT": "Weight [E]",
    "Smoker": "Smoker [F]",
    "PackYear": "Pack-year [G]",
    "ECOG": "ECOG [H]",
    "FVC": "FVC [I]",
    "FEV1": "FEV1 [J]",
    "DLCO": "DLCO-percent [K]",
    "DLCO_PERCENT": "DLCO-percent [L]",
    "SmallCell": "Squamous-cell-carcinoma [M]",
    "Aden": "Aden [N]",
    "LargeCell": "Large-cell [O]",
    "Positive_EGFR_Mutation": "Positive-EGFR-mutation [P]",
    "Positive_ALK_IHC": "Positive-ALK-IHC [Q]",
    "Positive_ALK_FISH": "Positive-ALK-FISH [R]",
    "STAGE_simple": "Stage-simple [S]",
    "OP_Curative": "OP-curative [T]",
    "RT": "RT [U]",
    "Chemo_curative": "Chemo-curative [V]",
    "Chemo_Palliative": "Chemo-palliative [W]",
}
new_columns = {
    "a_year_event": "A",
    "gender": "B",
    "age": "C",
    "HEIGHT": "D",
    "WEIGHT": "E",
    "Smoker": "F",
    "PackYear": "G",
    "ECOG": "H",
    "FVC": "I",
    "FEV1": "J",
    "DLCO": "K",
    "DLCO_PERCENT": "L",
    "SmallCell": "M",
    "Aden": "N",
    "LargeCell": "O",
    "Positive_EGFR_Mutation": "P",
    "Positive_ALK_IHC": "Q",
    "Positive_ALK_FISH": "R",
    "STAGE_simple": "S",
    "OP_Curative": "T",
    "RT": "U",
    "Chemo_curative": "V",
    "Chemo_Palliative": "W",
}
corr_df = corr_df.rename(index=new_index, columns=new_columns)

Path("plots").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(18, 9))

mask = np.triu(np.ones_like(corr_df, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

cmap.set_over('#8fd300')

heatmap = sns.heatmap(corr_df, vmin=0, vmax=0.999, annot=True, cmap=cmap, mask=mask)
plt.title(f"Cramer's V correlation")
plt.savefig(f"plots/correlation.png", dpi=300, bbox_inches='tight')