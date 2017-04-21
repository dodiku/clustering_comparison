'''
[ ] K-means, Hierarchical and Spectral techniques.
In particular, Hierarchical and Spectral methods should be extensively compared.

[ ] The accuracy of each clustering model should be assessed via test set,
for instance using the concentric noisy circles data presented during the lab class;

[ ] numerical study as to the impact of parameters in the quality of the clustering, mainly regarding spectral clustering, which demands several parameters.

[ ] Results should be presented as a report

'''

'''
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Dror Ayalon
'''

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
import csv
import pandas as pd
import clean

'''

X, Y = clean.clean()

## Kmeans
km = cluster.KMeans(n_clusters=2, random_state=0)
km.fit(X)
cl = km.labels_
plt.subplot(1,2,1)
plt.scatter(X[:,3],X[:,2],c=cl)

# Hierarchical
hc = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
hc.fit(X)
cl = hc.labels_
plt.subplot(1,2,2)
plt.scatter(X[:,3],X[:,2],c=cl)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=3)


Xspec = SpectralClustering(n_clusters = 2)
cl=Xspec.fit(X).labels_
#print(cl.shape)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=cl)


plt.subplot(2,3,3)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='arpack', affinity="nearest_neighbors", n_neighbors=20)
spectral_fit_predict = spectral.fit_predict(X)
plt.scatter(X.iloc[:,0],X.iloc[:,1], s=10, c=spectral_fit.labels_ ,cmap='rainbow')
plt.title("eigen_solver='arpack',\naffinity='nearest_neighbors',\nn_neighbors=20", fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.figure(1).set_size_inches(8,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.5)

plt.savefig('plots/spectral.png', dpi=300)
# plt.show()




svm_rbf.fit(x_training,y_training)
prediction_svm_rbf = svm_rbf.predict(x_test)
error_svm_rbf = np.sum((prediction_svm_rbf[i] != y_test[i]) for i in range(0,ts))

'''
'''------------------------------------
LOADING THE DATASET THAT INCLUDES THE FOLLOWING COLUMNS:
    [0] = Neighborhood
    [1] = BldClassif [classes]
    [2] = YearBuilt
    [3] = GrossSqFt
    [4] = GrossIncomeSqFt
    [5] = MarketValueperSqFt [data]
------------------------------------'''
X, Y = clean.clean()

'''--------------------
Spliting into test and validation groups
--------------------'''
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=3)
y_test = y_test.values - 1
y_train = y_train.values - 1
ts = len(y_test)


'''--------------------
Spectral clustering
# eigen_solver : {None, ‘arpack’, ‘lobpcg’, or ‘amg’}
# affinity : default ‘rbf’ or ‘nearest_neighbors’, ‘precomputed’
# n_neighbors
# assign_labels : {‘kmeans’, ‘discretize’}
--------------------'''

plt.figure(1).set_size_inches(6,4)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

plt.subplot(3,1,1)
plt.scatter(X.iloc[:,0],X.iloc[:,1], s=10, c=Y ,cmap='winter', alpha=0.6)
plt.title("Original Data", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,2)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=30)
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_model = spectral_fit.fit_predict(x_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
plt.scatter(x_train.iloc[:,0].values,x_train.iloc[:,1].values, s=10, c=spectral_fit_model ,cmap='rainbow', alpha=0.6)
plt.title("Spectral Clustering: Model", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,3)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=10, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("Spectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.savefig('plots/spectral.png', dpi=300)




plt.figure(2).set_size_inches(6,4)
plt.figure(2).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

plt.subplot(2,3,1)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='arpack', affinity='nearest_neighbors')
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='arpack', affinity='nearest_neighbors'\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.subplot(2,3,2)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=20)
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='arpack', affinity='nearest_neighbors'\nn_neighbors=20\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.subplot(2,3,3)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=30)
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='arpack', affinity='nearest_neighbors'\nn_neighbors=40\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.subplot(2,3,4)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='lobpcg', affinity='nearest_neighbors')
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='lobpcg', affinity='nearest_neighbors'\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.subplot(2,3,5)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='lobpcg', affinity='nearest_neighbors', n_neighbors=20)
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='lobpcg', affinity='nearest_neighbors'\nn_neighbors=20\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.subplot(2,3,6)
spectral = SpectralClustering(n_clusters = 2, eigen_solver='lobpcg', affinity='nearest_neighbors')
spectral_fit = spectral.fit(x_train, y_train)
spectral_fit_predict = spectral_fit.fit_predict(x_test)
error_spectral = 0
for i in range(0,ts):
    if spectral_fit_predict[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=4, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
plt.title("eigen_solver='lobpcg', affinity='nearest_neighbors'\nSpectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=6)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.savefig('plots/spectral_all.png', dpi=300)


'''--------------------
k-means
--------------------'''
plt.figure(3).set_size_inches(6,4)
plt.figure(3).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)


plt.subplot(3,1,1)
plt.scatter(X.iloc[:,0],X.iloc[:,1], s=10, c=Y ,cmap='winter', alpha=0.6)
plt.title("Original Data", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,2)
km = cluster.KMeans(n_clusters=2)
km = km.fit(x_train, y_train)
km_fit_model = km.fit_predict(x_train)
cl = km.predict(x_test)
plt.scatter(x_train.iloc[:,0].values,x_train.iloc[:,1].values, s=10, c=km_fit_model ,cmap='rainbow', alpha=0.6)
plt.title("K-means: Model", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,3)
error_spectral = 0
for i in range(0,ts):
    if cl[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=10, c=cl ,cmap='rainbow', alpha=0.6)
plt.title("K-means (Error: %s%%)" %(round(error_spectral,2)), fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.savefig('plots/k-means.png', dpi=300)

'''--------------------
Hierarchical
--------------------'''
plt.figure(4).set_size_inches(6,4)
plt.figure(4).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)


plt.subplot(3,1,1)
plt.scatter(X.iloc[:,0],X.iloc[:,1], s=10, c=Y ,cmap='winter', alpha=0.6)
plt.title("Original Data", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,2)
hc = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
hc = hc.fit(x_train, y_train)
hc_fit_model = hc.fit_predict(x_train)
cl = hc.fit_predict(x_test)
plt.scatter(x_train.iloc[:,0].values,x_train.iloc[:,1].values, s=10, c=hc_fit_model ,cmap='rainbow', alpha=0.6)
plt.title("Hierarchical: Model", fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.subplot(3,1,3)
error_spectral = 0
for i in range(0,ts):
    if cl[i] != y_test[i]:
        error_spectral = error_spectral + 1
error_spectral = error_spectral/ts*100
if error_spectral > 50:
    error_spectral = 100 - error_spectral
plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=10, c=cl ,cmap='rainbow', alpha=0.6)
plt.title("Hierarchical (Error: %s%%)" %(round(error_spectral,2)), fontsize=9)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.savefig('plots/Hierarchical.png', dpi=300)

# plt.show()

print ('\n~~~\n🐱  meow!\n~~~\n')
