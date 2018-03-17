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
from scipy.signal import lfilter
import scipy
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


N = X_train.shape[0]
# w = scipy.fftpack.rfft(y_train)
# print(N)
# print(w)
# plt.plot(range(1,N+1),w)
# # plt.plot(range(1,N+1),y_train)
# plt.show()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.plot(range(N), y_train,'o')
# plt.plot(range(N), smooth(y_train,3), 'r-', lw=2)
plt.plot(range(N), smooth(y_train,40), 'g-', lw=2)
plt.show()
