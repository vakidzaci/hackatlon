import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


#dateandtime  N EGT WF Status
df = pd.read_csv('../Dataset1.csv')
df.drop(['dateandtime'],inplace=True,axis=1)

# df =  df[df['N']!=" "]
for i in list(df):
    df[i].replace(" ", np.nan, inplace=True)
df = df.dropna()

for i in list(df):
    df[i] = pd.to_numeric(df[i])
# print df.corr()

# plt.scatter(df['N'],df['WF'])
# plt.show()

f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.show()
