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
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from scipy.signal import lfilter
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor

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


# err = []
# for i in range(1,100):
#     y_smooth = smooth(y_train,i)
#     reg.fit(X_train,y_smooth)
#     y_pred = reg.predict(X_test)
#     # l = X_test.shape[0]
#     # a = []
#     # for i in range(l):
#     #     a.append(i+1)
#     err.append(mean_squared_error(y_test,y_pred)**0.5)
# # plt.plot(a,y_test,color="red")
# # plt.plot(a,y_pred,color="blue")
# # plt.show()
# plt.plot(range(1,100),err)
# plt.show()

reg.fit(X,y)

march = pd.read_csv('march.csv')
days = []
months = []
years = []


march_date = np.array(march['date'])
for i in march_date:
    d = i.split('/')
    days.append(int(d[0]))
    months.append(int(d[1]))
    years.append(2018)

#
march['day'] = pd.DataFrame(days)
march['months'] = pd.DataFrame(months)
march['years'] = pd.DataFrame(years)
march.drop(['date'],inplace=True,axis=1)
#
print(reg.predict(march))
