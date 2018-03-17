import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

df = pd.read_csv('../Dataset3.csv')
date = np.array(df['dateandtime'])
df = df.iloc[::-1]

days = []
months = []
years = []

for i in date:
    d = i.split('/')
    days.append(int(d[0]))
    months.append(int(d[1]))
    years.append(int(d[2]))

df['day'] = pd.DataFrame(days)
df['months'] = pd.DataFrame(months)
df['years'] = pd.DataFrame(years)



reg = RandomForestRegressor()

y = df['Temperature_Almaty']
X = df.drop(['Temperature_Almaty','dateandtime'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=False)


model = RandomForestRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(X_test)
print(mean_squared_error(y_pred,y_test)**0.5)
l = X_test.shape[0]
a = []
for i in range(l):
    a.append(i+1)
plt.plot(a,y_test,color="red")
plt.plot(a,y_pred,color="blue")
plt.show()
