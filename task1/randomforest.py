import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

#dateandtime  N EGT WF Status
df = pd.read_csv('clean.csv')

y = df['Status']
X = df.drop(['Status'],axis=1)
model = RandomForestClassifier()
# for i in list(df):
    # X = np.array(df[i]).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# models = []
# print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))
print(cross_val_score(model, X, y, cv=10))
