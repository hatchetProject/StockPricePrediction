from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
import numpy as np
from train import series_to_supervised

def load_data():
    test_dataset = read_csv("train_data/test_data.csv", index_col=0)
    test_index = test_dataset.index.tolist()
    test_dataset = test_dataset.set_index("Date")
    test_dataset = test_dataset.drop("Time", axis=1)
    return test_dataset, test_index


def testing_dataset(dataset):
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled,scaler


def test_network(test_data, scaler):
    model = load_model("models/version2.h5")
    test_X = test_data
    """
    yhat = np.array([[[0]*20]])
    for i in range(test_data.shape[0]/10):
        tmp_data = test_data[10*i:10*(i+1)]
        tmp_data = tmp_data.reshape((1, 1, tmp_data.size))
        yhat_tmp = model.predict(tmp_data)
        yhat = concatenate((yhat, yhat_tmp), axis=0)
    yhat = yhat[1:]
    yhat = yhat.reshape((yhat.shape[0], yhat.shape[2]))
    yhat = yhat.reshape((yhat.size, 1))
    inv_yhat1 = concatenate((yhat[:yhat.size/2], test_data[:, 1:]), axis=1)
    inv_yhat1 = scaler.inverse_transform(inv_yhat1)
    inv_yhat1 = inv_yhat1[:,0]

    inv_yhat2 = concatenate((yhat[yhat.size/2:], test_data[:, 1:]), axis=1)
    inv_yhat2 = scaler.inverse_transform(inv_yhat2)
    inv_yhat2 = inv_yhat2[:, 0]
    inv_yhat = concatenate((inv_yhat1, inv_yhat2), axis=0)

    result = []
    for i in range(inv_yhat.shape[0]/20):
        tmp = 0
        for j in range(20):
            tmp += inv_yhat[j + 20*i]
        tmp /= 20
        result.append(tmp)
    return result
    """
    yhat = np.array([[0]])
    for i in range(test_data.shape[0]/10):
        tmp_data = test_data[10*i:10*(i+1)]
        tmp_data = tmp_data.reshape((1, 1, tmp_data.size))
        yhat_tmp = model.predict(tmp_data)
        yhat = concatenate((yhat, yhat_tmp), axis=0)
    yhat = yhat[1:]
    print (yhat.shape)
    print (test_X.shape)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
    test_X = test_X[:1000]
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    return inv_yhat

def write_csv(inv_yhat, index):
    df = pd.DataFrame()
    true_index = range(143, 1001)
    inv_yhat = inv_yhat[142:]
    df["caseid"] = true_index
    df["midprice"] = inv_yhat
    df.set_index("caseid")
    df.to_csv("version3.csv")

if __name__=="__main__":
    test_dataset, test_index = load_data()
    scaled_data, scaler = testing_dataset(test_dataset)
    yhat = test_network(scaled_data, scaler)
    write_csv(yhat, test_index)
