# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:19:27 2019

@author: HYDRA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=300, centers=5 , cluster_std=0.6)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans


wcb=[]

for i in range(1,15):
    km=KMeans(n_clusters= i)
    km.fit(x)
    wcb.append(km.inertia_)
    
plt.plot(range(1,15),wcb)
plt.show()
km= KMeans()
y_pred=km.predict(x)

plt.scatter(x[y_pred==0,0],x[y_pred==0,0])

plt.scatter(x[y_pred==0,1],x[y_pred==0,1])
plt.scatter(x[y_pred==0,2],x[y_pred==0,2])
plt.scatter(x[y_pred==0,3],x[y_pred==0,3])
plt.scatter(x[y_pred==0,4],x[y_pred==0,4])


