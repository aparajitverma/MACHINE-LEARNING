# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:21:50 2019

@author: HYDRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
dataset= load_iris()

X=dataset.data
y=dataset.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)