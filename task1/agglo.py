import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

import mpl_toolkits.mplot3d.axes3d as p3
from sklearn import preprocessing

df = pd.read_csv('clean.csv')
y = df['Status']
X = df.drop(['Status'],axis=1)
y = np.array(y)

X = preprocessing.scale(X)

ward = AgglomerativeClustering(
                n_clusters=3,
                affinity="cosine",
                memory=None,
                connectivity=None,
                compute_full_tree="auto",
                linkage="average",
                pooling_func=np.mean)
ward.fit(X)
label =np.array( ward.labels_)


print("Y")
unique, counts = np.unique(y, return_counts=True)
print np.asarray((unique, counts)).T

# print(y)
# print(label)
label = pd.DataFrame(label)
y = pd.DataFrame(y)

label = label.replace(0,11)
label = label.replace(1,0)
# label = label.replace(2,22)
print("LABEL")
unique, counts = np.unique(label, return_counts=True)
print np.asarray((unique, counts)).T


# y = y.replace(0,11)
y = y.replace(1,11)
# y = y.replace(2,22)
print(accuracy_score(y,label))
