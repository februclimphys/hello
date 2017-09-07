# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 11:23:11 2017

@author: mfebr
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


plt.close('all')

"""
In this part, we will read all text files and append it into different array
We put all the name of the data files that we want to read in "file list".
Read all the data files and put all the parameters into "all_data".
"""

file_list = []
fid = 'data1' # change this part to match the file name
alphabet = ('a','b','c','d','e')
for i in range(len(alphabet)): #change the increment
    file_list.append(fid+alphabet[i]+'.txt')

all_data = []
for file in file_list:
    pddata = pd.read_csv(file ,delim_whitespace=True,header=None, encoding='latin1')
    all_data.append(pddata)
    print ('reading '+str(file)+' done')
    
ts1 = np.zeros([len(all_data[0])]) #empty time series for data1a
ts2 = np.zeros([len(all_data[1])]) #empty time series for data1b
ts3 = np.zeros([len(all_data[2])]) #empty time series for data1c
ts4 = np.zeros([len(all_data[3])]) #empty time series for data1d
ts5 = np.zeros([len(all_data[4])]) #empty time series for data1e

ts1 = all_data[0].values
ts2 = all_data[1].values
ts3 = all_data[2].values
ts4 = all_data[3].values
ts5 = all_data[4].values

"""
The step for time-series analysis are:
    1) check for stationarity
    2) Calculate autocorrelation and partial correlation
    3) Identifying the order by tables
    4) recursive solution for the parameters (least square fittings)

"""


# Step 1: stationarity test
# Mean and variance test
# Time series 1
split_ts1 = int(len(ts1) / 2)
X1_ts1, X2_ts1 = ts1[0:split_ts1,0], ts1[split_ts1:,0]
mean1_ts1, mean2_ts1 = round(X1_ts1.mean(),2), round(X2_ts1.mean(),2)
var1_ts1, var2_ts1 = round(X1_ts1.var(),2), round(X2_ts1.var(),2)

# Time series 2
split_ts2 = int(len(ts2) / 2)
X1_ts2, X2_ts2 = ts2[0:split_ts2,0], ts2[split_ts2:,0]
mean1_ts2, mean2_ts2 = round(X1_ts2.mean(),2), round(X2_ts2.mean(),2)
var1_ts2, var2_ts2 = round(X1_ts2.var(),2), round(X2_ts2.var(),2)

# Time series 3
split_ts3 = int(len(ts3) / 2)
X1_ts3, X2_ts3 = ts3[0:split_ts3,0], ts3[split_ts3:,0]
mean1_ts3, mean2_ts3 = round(X1_ts3.mean(),2), round(X2_ts3.mean(),2)
var1_ts3, var2_ts3 = round(X1_ts3.var(),2), round(X2_ts3.var(),2)

# Time series 4
split_ts4 = int(len(ts4) / 2)
X1_ts4, X2_ts4 = ts4[0:split_ts4,0], ts4[split_ts4:,0]
mean1_ts4, mean2_ts4 = round(X1_ts4.mean(),2), round(X2_ts4.mean(),2)
var1_ts4, var2_ts4 = round(X1_ts4.var(),2), round(X2_ts4.var(),2)

# Time series 5
split_ts5 = int(len(ts5) / 2)
X1_ts5, X2_ts5 = ts5[0:split_ts5,0], ts5[split_ts5:,0]
mean1_ts5, mean2_ts5 = round(X1_ts5.mean(),2), round(X2_ts5.mean(),2)
var1_ts5, var2_ts5 = round(X1_ts5.var(),2), round(X2_ts5.var(),2)

# Step 2: autocorrelation and partial correlation
lag_acf_ts1 = acf(ts1[:,0], nlags = 20)
lag_pacf_ts1 = pacf(ts1[:,0], nlags = 20)
fig1 = plt.figure(1, figsize=(10,8), dpi=100)
ax1 = fig1.add_subplot(311)
ax1.set_title('Timeseries-1')
plt.plot(ts1[:,0])
plt.ylabel('amplitude')
plt.xlabel('N')

ax2 = fig1.add_subplot(312)
plt.plot(lag_acf_ts1)
ax2.set_title('autocorrelation')
plt.ylabel(r'$\rho$')
plt.xlabel('N-lags')

ax3 = fig1.add_subplot(313)
plt.plot(lag_pacf_ts1)
ax3.set_title(' partial autocorrelation')
plt.tight_layout()
plt.ylabel(r'P$\rho$')
plt.xlabel('N-lags')

plot_acf(ts1[:,0], lags = 50)
plot_pacf(ts1[:,0], lags = 50)

# ===================================

lag_acf_ts2 = acf(ts2[:,0], nlags = 20) #calculating the autocorrelation
lag_pacf_ts2 = pacf(ts2[:,0], nlags = 20) #calculating the partial - autocorrelation

fig2 = plt.figure(2, figsize=(10,8), dpi=100)
ax1 = fig2.add_subplot(311)
ax1.set_title('Timeseries-2')
plt.plot(ts2[:,0])
plt.ylabel('amplitude')
plt.xlabel('N')

ax2 = fig2.add_subplot(312)
plt.plot(lag_acf_ts2)
ax2.set_title('autocorrelation')
plt.ylabel(r'$\rho$')
plt.xlabel('N-lags')

ax3 = fig2.add_subplot(313)
plt.plot(lag_pacf_ts2)
ax3.set_title(' partial autocorrelation')
plt.ylabel(r'P$\rho$')
plt.xlabel('N-lags')
plt.tight_layout()

# ===================================

lag_acf_ts3 = acf(ts3[:,0], nlags = 20) #calculating the autocorrelation
lag_pacf_ts3 = pacf(ts3[:,0], nlags = 20) #calculating the partial - autocorrelation

fig3 = plt.figure(3, figsize=(10,8), dpi=100)
ax1 = fig3.add_subplot(311)
ax1.set_title('Timeseries-3')
plt.plot(ts3[:,0])
plt.ylabel('amplitude')
plt.xlabel('N')

ax2 = fig3.add_subplot(312)
plt.plot(lag_acf_ts3)
ax2.set_title('autocorrelation')
plt.ylabel(r'$\rho$')
plt.xlabel('N-lags')

ax3 = fig3.add_subplot(313)
plt.plot(lag_pacf_ts3)
ax3.set_title(' partial autocorrelation')
plt.ylabel(r'P$\rho$')
plt.xlabel('N-lags')
plt.tight_layout()

# ===================================

lag_acf_ts4 = acf(ts4[:,0], nlags = 20) #calculating the autocorrelation
lag_pacf_ts4 = pacf(ts4[:,0], nlags = 20) #calculating the partial - autocorrelation

fig4 = plt.figure(4, figsize=(10,8), dpi=100)
ax1 = fig4.add_subplot(311)
ax1.set_title('Timeseries-4')
plt.plot(ts4[:,0])
plt.ylabel('amplitude')
plt.xlabel('N')

ax2 = fig4.add_subplot(312)
plt.plot(lag_acf_ts4)
ax2.set_title('autocorrelation')
plt.ylabel(r'$\rho$')
plt.xlabel('N-lags')

ax3 = fig4.add_subplot(313)
plt.plot(lag_pacf_ts4)
ax3.set_title(' partial autocorrelation')
plt.ylabel(r'P$\rho$')
plt.xlabel('N-lags')
plt.tight_layout()


"""
Estimate model using ARIMA python
"""
arma_mod_ts1 = sm.tsa.ARMA(ts1[:,0], order=(1,1))
arma_res_ts1 = arma_mod_ts1.fit(trend='c', disp=-1)
print('AR summary ts-1')
print(arma_res_ts1.summary())




#max_lag = 20
#mdl = smt.AR(ts3[:,0]).fit(maxlag=max_lag, ic='aic', trend='c')
#est_order = smt.AR(ts1[:,0]).select_order(
#    maxlag=max_lag, ic='aic', trend='c')
#
#true_order = 2
#print ('\ncoef estimate: {:3.4f} {:3.4f} | best lag order = {}'
#  .format(mdl.params[0],mdl.params[1], est_order))

#model = ARIMA(ts1[:,0], order=(2, 1, 0))  
#results_AR = model.fit(disp=-1)  
#plt.plot(ts1[:,0])
#plt.plot(results_AR.fittedvalues, color='red')

testing