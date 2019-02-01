
# coding: utf-8

# In[349]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# In[350]:


dataset_train = pd.read_csv("train_data.csv")
Date = dataset_train.iloc[:,1:2].values
Time = dataset_train.iloc[:,2:3].values
MidPrice = dataset_train.iloc[:,3:4].values
LastPrice = dataset_train.iloc[:,4:5].values
Volume = dataset_train.iloc[:,5:6].values
BidPrice1 = dataset_train.iloc[:,6:7].values
BidVolume1 = dataset_train.iloc[:,7:8].values
AskPrice1 = dataset_train.iloc[:,8:9].values
AskVolume1 = dataset_train.iloc[:,9:10].values
print(Volume)
print(len(Volume))


# In[351]:


DVolume = Volume.copy()
for i in range(1,len(Volume)):
    DVolume[i] = Volume[i]- Volume[i-1]
DVolume[0] = DVolume[1]


# In[352]:


RTime = Time.copy()
timeArray = []
for i in range(len(Time)):
    time_s = Date[i][0] + ' ' + Time[i][0]
    timeArray.append(time.strptime(time_s,'%Y-%m-%d %H:%M:%S'))
print(timeArray)
for i in range(1,len(Time)):
    RTime[i] = int(time.mktime(timeArray[i]))
RTime[0] = RTime[1]-3
RTime


# In[271]:


from sklearn.preprocessing import scale
LastPrice_scaled = scale(LastPrice,axis=0, with_mean=True, with_std=True, copy=True)
BidPrice1_scaled = scale(BidPrice1,axis=0, with_mean=True, with_std=True, copy=True)
AskPrice1_scaled = scale(AskPrice1,axis=0, with_mean=True, with_std=True, copy=True)
DVolume_scaled = scale(DVolume,axis=0, with_mean=True, with_std=True, copy=True)
BidVolume1_scaled = scale(BidVolume1,axis=0, with_mean=True, with_std=True, copy=True)
AskVolume1_scaled = scale(AskVolume1,axis=0, with_mean=True, with_std=True, copy=True)
MidPrice_scaled = scale(MidPrice,axis=0, with_mean=True, with_std=True, copy=True)


# In[289]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
LastPrice_scaled = sc.fit_transform(LastPrice)
BidPrice1_scaled = sc.fit_transform(BidPrice1)
AskPrice1_scaled = sc.fit_transform(AskPrice1)
DVolume_scaled = sc.fit_transform(DVolume)
BidVolume1_scaled = sc.fit_transform(BidVolume1)
AskVolume1_scaled = sc.fit_transform(AskVolume1)
MidPrice_scaled = sc.fit_transform(MidPrice)
BidVolume1_scaled.max()


# In[353]:


def zscore(arr):
    u = arr.mean()
    sig = arr.std()
    res = np.copy(arr)
    for i in range(len(arr)):
        res[i] = (arr[i]-u)/sig
    return res,u,sig

def rev_zscore(arr,u,sig):
    res = np.copy(arr)
    for i in range(len(arr)):
        res[i] = arr[i]*sig + u
    return res


# In[354]:


LastPrice_scaled,LPu,LPsig = zscore(LastPrice)
BidPrice1_scaled,BP1u,BP1sig = zscore(BidPrice1)
AskPrice1_scaled,AP1u,AP1sig = zscore(AskPrice1)
DVolume_scaled,DVu,DVsig = zscore(DVolume)
BidVolume1_scaled,BV1u,BV1sig = zscore(BidVolume1)
AskVolume1_scaled,AV1u,AV1sig = zscore(AskVolume1)
MidPrice_scaled,MPu,MPsig = zscore(MidPrice)
BidVolume1_scaled.max()


# In[355]:


X_train = []
y_train = []
for i in range(10,len(DVolume_scaled)-20,1):
    if RTime[i+20] - RTime[i] != 60:
        continue
    tem_loader = np.array([])
    tem_loader = np.append(tem_loader,LastPrice_scaled[i-10:i,0])
    tem_loader = np.append(tem_loader,DVolume_scaled[i-10:i,0])
    tem_loader = np.append(tem_loader,BidVolume1_scaled[i-10:i,0])
    tem_loader = np.append(tem_loader,AskVolume1_scaled[i-10:i,0])
    tem_loader = np.append(tem_loader,BidPrice1_scaled[i-10:i,0])
    tem_loader = np.append(tem_loader,AskPrice1_scaled[i-10:i,0])
    X_train.append(tem_loader)
    y_train.append(MidPrice_scaled[i:i+20,0].mean())
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
print(X_train)
print(y_train)
print(X_train.shape[0],X_train.shape[1])
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
print(len(X_train),len(y_train))


# In[356]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model


# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units=128,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=128,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=128))
regressor.add(Dropout(0.2))
regressor.add(Dense(128,activation='relu'))   
regressor.add(Dense(128,activation='relu'))
regressor.add(Dense(units=1))
regressor.compile(optimizer = 'Adam',loss='mean_squared_error')
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


# In[342]:


regressor.save('stock_price_v16_100e_zscore.h5')


# In[273]:


regressor = load_model("stock_price_v10_norm.h5")


# In[343]:


dataset_test = pd.read_csv('test_data.csv')
MidPrice_test = dataset_test.iloc[:,3:4].values
LastPrice_test = dataset_test.iloc[:,4:5].values
Volume_test = dataset_test.iloc[:,5:6].values
BidPrice1_test = dataset_test.iloc[:,6:7].values
BidVolume1_test = dataset_test.iloc[:,7:8].values
AskPrice1_test = dataset_test.iloc[:,8:9].values
AskVolume1_test = dataset_test.iloc[:,9:10].values
Volume_test


# In[344]:


DVolume_test = Volume_test.copy()
for i in range(1,len(Volume_test)):
    DVolume_test[i] = Volume_test[i]- Volume_test[i-1]
DVolume_test[0] = DVolume_test[1]
DVolume_test


# In[276]:


LastPrice_test_scaled = scale(LastPrice_test,axis=0, with_mean=True, with_std=True, copy=True)
BidPrice1_test_scaled = scale(BidPrice1_test,axis=0, with_mean=True, with_std=True, copy=True)
AskPrice1_test_scaled = scale(AskPrice1_test,axis=0, with_mean=True, with_std=True, copy=True)
DVolume_test_scaled = scale(DVolume_test,axis=0, with_mean=True, with_std=True, copy=True)
BidVolume1_test_scaled = scale(BidVolume1_test,axis=0, with_mean=True, with_std=True, copy=True)
AskVolume1_test_scaled = scale(AskVolume1_test,axis=0, with_mean=True, with_std=True, copy=True)
MidPrice_test_scaled = scale(MidPrice_test,axis=0, with_mean=True, with_std=True, copy=True)


# In[331]:


DVolume_test_scaled = sc.fit_transform(DVolume_test)
BidVolume1_test_scaled = sc.fit_transform(BidVolume1_test)
AskVolume1_test_scaled = sc.fit_transform(AskVolume1_test)
LastPrice_test_scaled = sc.fit_transform(LastPrice_test)
BidPrice1_test_scaled = sc.fit_transform(BidPrice1_test)
AskPrice1_test_scaled = sc.fit_transform(AskPrice1_test)
BidVolume1_test_scaled.mean()


# In[345]:


LastPrice_test_scaled,LPu,LPsig = zscore(LastPrice_test)
BidPrice1_test_scaled,BP1u,BP1sig = zscore(BidPrice1_test)
AskPrice1_test_scaled,AP1u,AP1sig = zscore(AskPrice1_test)
DVolume_test_scaled,DVu,DVsig = zscore(DVolume_test)
BidVolume1_test_scaled,BV1u,BV1sig = zscore(BidVolume1_test)
AskVolume1_test_scaled,AV1u,AV1sig = zscore(AskVolume1_test)
MidPrice_test_scaled,MPu,MPsig = zscore(MidPrice_test)


# In[346]:


X_test = []

for i in range(len(MidPrice_test) // 10):
    tem_loader = np.array([])
    tem_loader = np.append(tem_loader,LastPrice_test_scaled[i*10:(i+1)*10])
    tem_loader = np.append(tem_loader,DVolume_test_scaled[i*10:(i+1)*10])
    tem_loader = np.append(tem_loader,BidVolume1_test_scaled[i*10:(i+1)*10])
    tem_loader = np.append(tem_loader,AskVolume1_test_scaled[i*10:(i+1)*10])
    tem_loader = np.append(tem_loader,BidPrice1_test_scaled[i*10:(i+1)*10])
    tem_loader = np.append(tem_loader,AskPrice1_test_scaled[i*10:(i+1)*10])
    X_test.append(tem_loader)

X_test = np.array(X_test)

X_test


# In[347]:


X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
print(X_test)
y_test = regressor.predict(X_test)
#y_test = sc.inverse_transform(y_test)
y_test = rev_zscore(y_test,MPu,MPsig)
y_test


# In[348]:



num_id = []
mid_price = []
for i,j in enumerate(y_test):
    num_id.append(i)
    mid_price.append(j[0])
datafm = pd.DataFrame({'mid_price':mid_price})
print(datafm)
datafm.to_csv('res0.csv',index = False,sep = ' ')


# In[ ]:


plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')

plt.title('Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()


# In[22]:


regressor.save("stock_price_v1.h5")

