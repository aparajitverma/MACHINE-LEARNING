# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:30:14 2019

@author: HYDRA
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('bank.csv',sep=';')
                    
X=dataset.iloc[:,0:16].values
y=dataset.iloc[:,-1].values
from sklearn.impute import SimpleImputer
sim=SimpleImputer()
X[:,[0,5,9,11,12,13,14]]=sim.fit_transform(X[:,[0,5,9,11,12,13,14]])
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
X[:, 1]= lab.fit_transform(X[:, 1])
X[:, 2]= lab.fit_transform(X[:, 2])
X[:, 3]= lab.fit_transform(X[:, 3])
X[:, 4]= lab.fit_transform(X[:, 4])
X[:, 6]= lab.fit_transform(X[:, 6])
X[:, 7]= lab.fit_transform(X[:, 7])

X[:, 8]= lab.fit_transform(X[:, 8])
X[:, 10]= lab.fit_transform(X[:, 10])
X[:, 15]= lab.fit_transform(X[:, 15])

from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features= [1,2,3,4,6,7,8,10,15])
X=one.fit_transform(X)
X=X.toarray()
y=lab.fit_transform(y)
lab.classes_
        #new code
#y=y.reshape(-1,1)

#plt.scatter(X,y)
#plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2, include_bias='false')

X_poly=poly.fit_transform(X)

#poly.score(X,y)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
#y_new= lin_reg.predict(X)
lin_reg.score(X,y)      #29%

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)





from sklearn.linear_model import LogisticRegression
log =LogisticRegression()
log.fit(X,y)
log.score(X,y)
#90%






