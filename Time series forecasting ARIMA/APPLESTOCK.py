# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:37:06 2019

@author: HYDRA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller #for finding unit root   p
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf




datasets=pd.read_csv('AAPL.csv')




X=datasets.iloc[:, 0].values
y=datasets.iloc[:, 1:7].values
plt.scatter(X,y[:,0])

plt.scatter(X,y[:,1], label='Open')

plt.scatter(X,y[:,2])

plt.scatter(X,y[:,3])

plt.scatter(X,y[:,4])

plt.scatter(X,y[:,5])

plt.plot(X,y[:,0], label='Open')
plt.plot(X,y[:,1],label='High')
plt.plot(X,y[:,2],label='Low')
plt.plot(X,y[:,3],label='Close')
plt.plot(X,y[:,4],label='Adj Close')
plt.plot(X,y[:,5],label='Volume')

plt.show()




result= adfuller(datasets.Close.dropna())


print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])        #p value greater than 0.05 hence diffrecing needs to be done


#Original series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(datasets.Close); axes[0, 0].set_title('Original Series')
plot_acf(datasets.Close, ax=axes[0, 1])
# 1st Differencing

axes[1, 0].plot(datasets.Close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(datasets.Close.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(datasets.Close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(datasets.Close.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

y1=datasets.Close 
#Perform a test of stationarity for different levels of ``d`` to estimate  the number of differences
# required to make a given time series stationary

ndiffs(y1,test= 'adf')

ndiffs(y1,test= 'kpss')

ndiffs(y1,test= 'pp')
#result=1,1,1

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(datasets.Close.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(datasets.Close.diff().dropna(), ax=axes[1])
#giving the value of AR part or p as 1 from result
plt.show()


fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(datasets.Close.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(datasets.Close.diff().dropna(), ax=axes[1])

plt.show()

#giving the value of MA part or q as 1 from result

model= ARIMA(datasets.Close,order=(1,1,1))# applying algorithm

model_fit= model.fit(disp=0) #fitting with the dataset
print(model_fit.summary())

#looking for patterns in error(constant mean and variance)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
#zero mean and uniform variance


#OUT OF TIME CROSS VALIDATION
# Create Training and Test
train = datasets.Close[:60]
test = datasets.Close[38:]



model = ARIMA(train, order=(2, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(146, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], #stacking arrays n sequence horizontally
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

# accuracy 85.7%


pred = model.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()



























