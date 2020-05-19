#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:39:26 2020

@author: abbychen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

######## data input ############
data=pd.read_csv("diabetes.csv") 
data.isnull().sum()
data.isna().sum()
data.columns

data.hist(figsize=(20,20))
data.shape

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn')

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values


####### standard scale ########
sc=StandardScaler()
X=sc.fit_transform(X)

######### Outlier ##########

outliers_fraction = 0.01
outlier_model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
outlier_model.fit(X)
out = outlier_model.predict(X)


df = pd.DataFrame({'out_prediction':out})
df=df[df['out_prediction']==1]

b=set(df.index.values.tolist())
a=[]
for i in range(0,len(X)):
    if i in b:
        a.append(X[i])
X=np.array(a)   

c=[]
for i in range(0,len(Y)):
    if i in b:
        c.append(Y[i])
Y=np.array(c)     
    

##########PCA##############


pca = PCA()
X=pca.fit_transform(X)

vt = VarianceThreshold()
vt.fit(X)
plt.figure(figsize=(4,3))
plt.plot(np.arange(0,len(X[0])),pd.Series(vt.variances_), color=sns.color_palette('colorblind')[1])
plt.xlabel('Principle component #'); 
plt.ylabel('Variance in component dimension');
plt.title('Principle component variances before thresholding', size=14);
vt1= VarianceThreshold(threshold=0.6)
X = vt1.fit_transform(X)



########## Split ##############
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.25)

######### Model #######
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2 )
classifier.fit(X_train,Y_train)
classifier.score(X_train,Y_train)

Y_pred=classifier.predict(X_test)
classifier.score(X_test,Y_test)

#######confusion matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy_score(Y_test,Y_pred)

from sklearn.model_selection import GridSearchCV
parameter={'n_neighbors':[1,2,3,5,6,7,8,9,10,11,12,13],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'metric':['euclidean','manhattan','minkowski']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameter,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)



rfc=RandomForestClassifier(max_depth = 2 )
rfc.fit(X_train,Y_train)
rfc.score(X_train,Y_train)

Y_pred=rfc.predict(X_test)
rfc.score(X_test,Y_test)

#######confusion matrix
cm=confusion_matrix(Y_test,Y_pred)
print(cm)
accuracy_score(Y_test,Y_pred)











