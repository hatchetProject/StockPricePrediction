import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import concatenate
from math import sqrt
from train import series_to_supervised
import time

class Random_Forest_Regressor:
    def __init__(self, n_estimate=10, depth=3,  crit="mse", vb=0, random=0):
        self.train_set = None
        self.train_label = None
        self.validation_set = None
        self.validation_label = None
        self.test_set = None
        self.model = RandomForestRegressor(n_estimators=n_estimate, max_depth=depth, criterion=crit, verbose=vb, random_state=random)

    def data_clip(self, features, labels):
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.1, random_state = 0)
        return x_train, x_valid, y_train, y_valid

    def train(self, features, labels):
        self.train_set, self.validation_set, self.train_label, self.validation_label = self.data_clip(features, labels)
        self.model.fit(self.train_set, self.train_label)
        predict = self.model.predict(self.validation_set)
        mse = mean_squared_error(predict, self.validation_label)
        print ("MSE: ", mse)
        return mse

    def test(self, features_test, scaler):
        yhat = []
        for test_data in features_test:
            test_data = test_data.reshape((1, -1))
            test_prediction = self.model.predict(test_data)
            yhat.append(test_prediction)
        yhat = np.array(yhat)
        yhat = yhat.reshape((1000, 1))
        test_X = features_test.reshape((10000, 7))
        test_X = test_X[:1000]
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        return inv_yhat


def load_data():
    train_dataset = read_csv("train_data/train_data.csv", index_col=0)
    return train_dataset


def clip_data_by_time(dataset_train, RTime):
    drop_rows = []

    for i in range(10, len(RTime)-20, 1):
        if (RTime[i+20] - RTime[i] > 65):
            drop_rows.append(i)
    dataset_train = dataset_train.drop(drop_rows)
    return dataset_train


def cs_to_sl(dataset, n_in, n_out):
    # load dataset
    Date = dataset.iloc[:,0:1].values
    Time = dataset.iloc[:,1:2].values
    RTime = Time.copy()
    timeArray = []
    for i in range(len(Time)):
        time_s = Date[i][0] + ' ' + Time[i][0]
        timeArray.append(time.strptime(time_s,'%Y-%m-%d %H:%M:%S'))
    for i in range(1,len(Time)):
        RTime[i] = int(time.mktime(timeArray[i]))
    RTime[0] = RTime[1]-3
    RTime = RTime.reshape((RTime.size, ))
    dataset = dataset.drop(["Time", "Date"], axis=1)
    #dataset_train["time"] = RTime

    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)

    reframed = reframed.drop((("var%d(t)" % (j)) for j in range(2, 8)) , axis=1)
    for i in range(n_out-1):
        reframed = reframed.drop((("var%d(t+%d)" % (j, i+1)) for j in range(2, 8)) , axis=1)

    return reframed,scaler, RTime


def training_dataset(reframed):
    sum_20 = reframed[["var1(t)", "var1(t+1)", "var1(t+2)", "var1(t+3)", "var1(t+4)", "var1(t+5)", "var1(t+6)", "var1(t+7)", "var1(t+8)",
                        "var1(t+9)", "var1(t+10)", "var1(t+11)", "var1(t+12)", "var1(t+13)", "var1(t+14)", "var1(t+15)", "var1(t+16)",
                        "var1(t+17)", "var1(t+18)", "var1(t+19)"]]
    sum_20["col_sum"] = sum_20.apply(lambda x: x.sum()/20, axis=1)

    reframed = reframed.drop(["var1(t)", "var1(t+1)", "var1(t+2)", "var1(t+3)", "var1(t+4)", "var1(t+5)", "var1(t+6)", "var1(t+7)", "var1(t+8)",
                        "var1(t+9)", "var1(t+10)", "var1(t+11)", "var1(t+12)", "var1(t+13)", "var1(t+14)", "var1(t+15)", "var1(t+16)",
                        "var1(t+17)", "var1(t+18)", "var1(t+19)"], axis=1)
    reframed["var1(t)"] = sum_20["col_sum"]
    train_data = reframed.values
    n_train_time = reframed.shape[0]
    train_data = train_data[:n_train_time, :]
    train_X, train_y = train_data[:, :-1], train_data[:, -1:]
    train_y = train_y.reshape((train_y.size, ))
    print (train_X.shape)
    print (train_y.shape)

    return train_X,train_y


def load_test_data():
    test_dataset = read_csv("train_data/test_data.csv", index_col=0)
    test_dataset = test_dataset.set_index("Date")
    test_dataset = test_dataset.drop("Time", axis=1)
    return test_dataset

def testing_dataset(dataset):
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled,scaler


def write_csv(inv_yhat):
    df = pd.DataFrame()
    true_index = range(143, 1001)
    inv_yhat = inv_yhat[142:]
    df["caseid"] = true_index
    df["midprice"] = inv_yhat
    df.set_index("caseid", inplace=True)
    df.to_csv("RF.csv")


if __name__=="__main__":
    train_dataset = load_data()

    reframed_train, _, RTime = cs_to_sl(train_dataset, 10, 20)
    train_dataset = clip_data_by_time(train_dataset, RTime)
    train_x,train_y = training_dataset(reframed_train)

    RF_Model = Random_Forest_Regressor(50, 10)
    print ("Begin Training")
    mse_loss = RF_Model.train(train_x, train_y)
    print ("Training Finished")
    test_dataset =load_test_data()
    scaled_dataset, scaler = testing_dataset(test_dataset)
    scaled_dataset = scaled_dataset.reshape((1000, 70))
    test_result = RF_Model.test(scaled_dataset, scaler)
    write_csv(test_result)
