from matplotlib import pyplot
import plotly.graph_objects as go
from scipy.signal import find_peaks
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


data_coup=pd.read_excel('/Users/enkhbat/fiverr/nurez_damani/nurez_damani-attachments/Statistical Analysis - CIM.xlsx')


# split a dataset into train/test sets
def train_test_split(data, n_test):
  return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
  df = pd.DataFrame(data)
  cols = list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
  # put it all together
  agg = concat(cols, axis=1)
  # drop rows with NaN values
  agg.dropna(inplace=True)
  return agg.values

 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
  return sqrt(mean_squared_error(actual, predicted))
 
# difference dataset
def difference(data, order):
  return [data[i] - data[i - order] for i in range(order, len(data))]
 
# fit a model
def model_fit(train, config):
  # unpack config
  n_input, n_nodes, n_epochs, n_batch, n_diff = config
  # prepare data
  if n_diff > 0:
    train = difference(train, n_diff)
  # transform series into supervised format
  data = series_to_supervised(train, n_in=n_input)
  # separate inputs and outputs
  train_x, train_y = data[:, :-1], data[:, -1]
  # define model
  model = Sequential()
  model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  # fit model
  model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
  return model
 
# forecast with the fit model
def model_predict(model, history, config):
  # unpack config
  n_input, _, _, _, n_diff = config
  # prepare data
  correction = 0.0
  if n_diff > 0:
    correction = history[-n_diff]
    history = difference(history, n_diff)
  # shape input for model
  x_input = array(history[-n_input:]).reshape((1, n_input))
  # make forecast
  yhat = model.predict(x_input, verbose=0)
  # correct forecast if it was differenced
  return correction + yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
  predictions = list()
  # split dataset
  train, test = train_test_split(data, n_test)
  # fit model
  model = model_fit(train, cfg)
  # seed history with training dataset
  history = [x for x in train]
  # step over each time-step in the test set
  for i in range(len(test)):
    # fit model and make forecast for history
    yhat = model_predict(model, history, cfg)
    # store forecast in list of predictions
    predictions.append(yhat)
    # add actual observation to history for the next loop
    history.append(test[i])
  # estimate prediction error
  error = measure_rmse(test, predictions)
  print(' > %.3f' % error)
  return error, predictions, test

# Formula for long return
def long(x):
  long_ret=(df['Price '][0]-df['Price '][x])/df['Price '][x]
  return long_ret
# Formula for short return
def short(x):
  short_ret=(df['Price '][0]-df['Price '][x])/df['Price '][0]
  return short_ret
# Cut the first two useless rows
def excel_to_df(df):
  df=df.iloc[1:,0:2]
  # make rows to columns
  df.columns = df.iloc[0]
  # drop index
  df=df.drop(df.index[0])
  # reset index
  df=df.reset_index()
  return df

def time_series_long(df):
  df['RETURNS LONG']=df.index.map(lambda x:long(x))
  return df

def time_series_short(df):
  df['RETURNS SHORT']=df.index.map(lambda x:short(x))
  return df


# This price vs time dataset

def long_to_csv(df):
  df=excel_to_df(df)
  df=time_series_long(df)
  df=df[['RETURNS LONG','Time']]
  df['Time'] = pd.to_datetime(df['Time'])
  df.Time=df.Time.map(lambda x:str(x))
  df=df.set_index('Time')
  return df.to_csv('long_term_ret.csv')


def short_to_csv(df):
  df=excel_to_df(df)
  df=time_series_short(df)
  df=df[['RETURNS SHORT','Time']]
  df['Time'] = pd.to_datetime(df['Time'])
  df.Time=df.Time.map(lambda x:str(x))
  df=df.set_index('Time')
  return df.to_csv('short_term_ret.csv')



df=excel_to_df(data_coup)

long_to_csv(data_coup)
short_to_csv(data_coup)

# define dataset
series = pd.read_csv('long_term_ret.csv', header=0, index_col=0)
# series['Pass']=series['Passengers']
data = series.values
# data split
n_test = 12
# model configs
cfg=[20, 50, 100, 1, 0] 

true_values=walk_forward_validation(data[:110], n_test, cfg)[1]
predictions=walk_forward_validation(data[:110], n_test, cfg)[0]

def plot_it(true_values, predictions):
  # Create plots with pre-defined labels.
  fig, ax = plt.subplots()
  ax.plot([x for x in true_values], label="True val")
  ax.plot([x for x in predictions], label="Pred val")
  ax.set_title('True values vs Predicted values')
  ax.set_ylabel('Ruturn Long')
  ax.set_xlabel('Number of Observation')
  ax.legend()
  plt.show()

  plot_it(true_values, predictions)