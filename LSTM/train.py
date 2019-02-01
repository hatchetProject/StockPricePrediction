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
from sklearn.preprocessing import scale
import time

def parse_time(x):
    return datetime.strptime(x, '%Y %m %d %H')


def load_data():
    train_dataset = read_csv("train_data/train_data.csv", index_col=0)
    return train_dataset

def clip_data_by_time(dataset_train):
    Date = dataset_train.iloc[:,0:1].values
    Time = dataset_train.iloc[:,1:2].values
    RTime = Time.copy()
    timeArray = []
    for i in range(len(Time)):
        time_s = Date[i][0] + ' ' + Time[i][0]
        timeArray.append(time.strptime(time_s,'%Y-%m-%d %H:%M:%S'))
    for i in range(1,len(Time)):
        RTime[i] = int(time.mktime(timeArray[i]))
    RTime[0] = RTime[1]-3
    RTime = RTime.reshape((RTime.size, ))
    dataset_train = dataset_train.drop(["Time", "Date"], axis=1)
    dataset_train["time"] = RTime
    drop_rows = []

    for i in range(10, len(RTime)-20, 1):
        if (RTime[i+20] - RTime[i] > 65):
            drop_rows.append(i)
    dataset_train = dataset_train.drop(drop_rows)
    dataset_train = dataset_train.drop("time", axis=1)
    return dataset_train

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
     # convert series to supervised learning
    	n_vars = 1 if type(data) is list else data.shape[1]
    	df = pd.DataFrame(data)
    	cols, names = list(), list()
        names_index = df.columns.values.tolist()
    	# input sequence (t-n, ... t-1)
    	for i in range(n_in, 0, -1):
    		cols.append(df.shift(i))
    		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    	# forecast sequence (t, t+1, ... t+n)
    	for i in range(0, n_out):
    		cols.append(df.shift(-i))
    		if i == 0:
    			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    		else:
    			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	# put it all together
    	agg = pd.concat(cols, axis=1)
    	agg.columns = names
    	# drop rows with NaN values
    	if dropnan:
    		agg.dropna(inplace=True)
    	return agg


def plot_feature():
    dataset = load_data()
    values = dataset.values
    groups = range(6)
    i = 1
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc="right")
        i += 1
    pyplot.show()


def cs_to_sl(dataset, n_in, n_out):
    # load dataset
    values = dataset.values
    # integer encode direction
    # This step is for transforming non-number data type into labels, such as
    # turning East into 1, North into 2 etc.
    #encoder = LabelEncoder()
    #values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    # drop columns we don't want to predict

    reframed = reframed.drop((("var%d(t)" % (j)) for j in range(2, 8)) , axis=1)
    for i in range(n_out-1):
        reframed = reframed.drop((("var%d(t+%d)" % (j, i+1)) for j in range(2, 8)) , axis=1)
    #print(reframed)
    print(reframed.head())
    return reframed,scaler


def training_dataset(reframed):
    # Trained with the previous midprice, which means the next timestamp's midprice
    # depends on the previous timestamp's midprice
    # Could try without it later, just drop the first column
    # split into train and test sets
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
    print (train_X.shape)
    print (train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    #train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    #print (train_y.shape)
    return train_X,train_y


def fit_network(train_X,train_y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_split = 0.20, verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    model.save("models/version3.h5")


if __name__ == '__main__':
    #plot_feature()
    train_dataset = load_data()
    train_dataset = clip_data_by_time(train_dataset)
    reframed_train, _ = cs_to_sl(train_dataset, 10, 20)

    train_X,train_y = training_dataset(reframed_train)
    fit_network(train_X,train_y)
