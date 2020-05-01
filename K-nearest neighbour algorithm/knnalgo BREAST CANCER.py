# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:13:18 2019

@author: HYDRA
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
datasets=load_breast_cancer()
X=datasets.data
y=datasets.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X,y)