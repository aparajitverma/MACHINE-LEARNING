# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:01:33 2019

@author: HYDRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('train.csv')
datasettest=pd.read_csv('test.csv')
X=dataset.iloc[:, 2:11].values
y=dataset.iloc[:, 1].values


from sklearn.impute import SimpleImputer
sim= SimpleImputer()

X[:,3]= sim.fit_transform(X[:,3])

temp= pd.DataFrame(X[:,8])
temp[0].value_counts()

temp[0] = temp[0].fillna(' G6')

X[:, 5] = temp[0]

del(temp)

from sklearn.preprocessing import LabelEncoder
lab= LabelEncoder()
X[:, 1]= lab.fit_transform(X[:, 1])

X[:, 2]= lab.fit_transform(X[:, 2])

X[:, 6]= lab.fit_transform(X[:, 6])


X[:, 8]= lab.fit_transform(X[:, 8])

X[:, 9]= lab.fit_transform(X[:, 9])

lab.classes_



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier(max_depth=13)
dtf.fit(X_train,y_train)
y_new=dtf.predict(X_test)
dtf.score(X_train,y_train)




