# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:24:25 2019

@author: HYDRA
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

from statsmodels.tsa.stattools import acf, pacf

import tensorflow
import keras


df = pd.read_excel("BA_Combined.xlsx",parse_dates=[['Year','Month','Day','Hour','Minute']])

df['Year_Month_Day_Hour_Minute'] = pd.to_datetime(df.Year_Month_Day_Hour_Minute , format = '%Y %m %d %H %M')
#new=df[df['GHI']!=0]
df1= df[:2000]
#test1= df[:1200]
exog= df1.iloc[:,[1,2,4,5,6,7,8,9]].values






#test['Year_Month_Day_Hour_Minute'].min(), test['Year_Month_Day_Hour_Minute'].max()

#t#est.isnull().sum()

#test = test.groupby('Year_Month_Day_Hour_Minute')['GHI'].sum().reset_index()

df1 = df1.set_index('Year_Month_Day_Hour_Minute')
df1.index


y = df1['GHI']#.resample('MS').mean()
y= y[:2000]
















from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
#split into test and train
percentage = 0.6
series = df1['GHI'].tolist()
size = int(len(df1) * 0.66)
train, test = y[0:size], y[size:len(series)]
#model = ARIMA(train , order = (1,0,0))
#model_fit = model.fit()

from statsmodels.tsa.stattools import acf, pacf
acf_1 = acf(y)[1:20]
plt.plot(acf_1)
test_df = pd.DataFrame([acf_1]).T
test_df.columns = ["Pandas Autocorrelation"]
test_df.index += 1
test_df.plot(kind='bar')
pacf_1 = pacf(y)[1:20]
plt.plot(pacf_1)
plt.show()
test_df = pd.DataFrame([pacf_1]).T
test_df.columns = ['Pandas Partial Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.recurrent import LSTM
history = train


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            
            results = mod.fit()
            
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            mod = sm.tsa.statespace.SARIMAX(train,exogenous=exog,
                                order=(0, 0, 0),
                                seasonal_order=(0, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    continue

pred = results.get_prediction(start=1320,end=1999,dynamic=True, full_results=True)


pred_ci = pred.conf_int()
ax = train['2010-01-01 00:30:00 ':].plot(label='observed',color='green')
predict=pred.predicted_mean
predict1=predict
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.5)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='b', alpha=.5)
ax.set_xlabel('Date')
ax.set_ylabel('GHI')
plt.legend()
plt.show()


forecasted = pred.predicted_mean
pred_uc = results.get_forecast(steps=500)
pred_ci = pred_uc.conf_int()
ax = train.plot(label='observed', figsize=(14, 7))
predict.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Close')
plt.legend()
plt.show()





def forecast_accuracy(forecast, actual):
    maape =np.mean(np.arctan(np.abs((actual - forecast) / (actual))))
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
                   # ACF1
    return({'maape':maape,'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(predict, test)

resid_test = []


for t in range(len(test)):
    resid_test.append(test[t] - forecasted[0])
























test_resid = []
for i in resid_test:
    test_resid.append(i)
error = mean_squared_error(test, forecasted)
print('Test MSE: %.3f' % error)
plt.plot(test)
plt.plot(predict, color='red')
plt.show()


print(results.summary())
# plot residual errors
residuals = pd.DataFrame(results.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
#plot the acf for the residuals
acf_1 = acf(results.resid)[1:20]
plt.plot(acf_1)
test_df = pd.DataFrame([acf_1]).T
test_df.columns = ["Pandas Autocorrelation"]
test_df.index += 1
test_df.plot(kind='bar')
#HYBRID

window_size = 50
rmse =[]



def make_model(window_size):
    model = Sequential()
    model.add(Dense(50, input_dim=window_size, init="uniform",
    activation="tanh"))
    model.add(Dense(25, init="uniform", activation="relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = make_model(50)
#lstm_model = make_lstm_model()
min_max_scaler = preprocessing.MinMaxScaler()
train = np.array(train).reshape(-1,1)


test_extended = train.tolist()[-1*window_size:] + test_resid 
test_data = []
for i in test_extended:
    try:
        test_data.append(i[0])
    except:
        test_data.append(i)
test_data = np.array(test_data).reshape(-1,1)











train_scaled = min_max_scaler.fit_transform(test_data) 

train_X,train_Y = [],[]
for i in range(0 , len(train_scaled) - window_size):
    train_X.append(train_scaled[i:i+window_size])
    train_Y.append(train_scaled[i+window_size])

new_train_X,new_train_Y = [],[]
for i in train_X:
    new_train_X.append(i.reshape(-1))
for i in train_Y:
    new_train_Y.append(i.reshape(-1))
new_train_X = np.array(new_train_X)
new_train_Y = np.array(new_train_Y)



predict1=np.array(predict1).reshape(-1,1)




#new_train_X = np.reshape(new_train_X, (new_train_X.shape[0], new_train_X.shape[1], 1))
model.fit(new_train_X,new_train_Y, nb_epoch=500, batch_size=512, validation_split = .05)



test_extended = train.tolist()[-1*window_size:] + test_resid 
min_max_scaler = preprocessing.MinMaxScaler()
test_scaled = min_max_scaler.fit_transform(test_data)
test_X,test_Y = [],[]
for i in range(0 , len(test_scaled) - window_size):
    test_X.append(test_scaled[i:i+window_size])
    test_Y.append(test_scaled[i+window_size])
    new_test_X,new_test_Y = [],[]
for i in test_X:
    new_test_X.append(i.reshape(-1))
for i in test_Y:
    new_test_Y.append(i.reshape(-1))
new_test_X = np.array(new_test_X)
new_test_Y = np.array(new_test_Y)
#new_test_X = np.reshape(new_test_X, (new_test_X.shape[0], new_test_X.shape[1], 1))
predictions = model.predict(new_train_X)
predictions_rescaled=min_max_scaler.inverse_transform(predictions)
Y = pd.DataFrame(new_train_Y)
pred = pd.DataFrame(predictions)
plt.plot(Y)
plt.plot(pred , color = 'r')
#p.plot()
plt.show()
test_resid= np.array(test_resid).reshape(-1,1)
error = mean_squared_error(test_resid,predictions_rescaled)
print('Test MSE: %.3f' % error)
forecast_accuracy(test_resid,predictions_rescaled)



train








