import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from pandas_datareader import DataReader
import yfinance as yf

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','KTKBANK.NS')
df = yf.download(user_input,period='60mo')

# describing data

st.subheader('Data from last 60 months')
st.write(df.describe())

# visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with  100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with  100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array=scaler.fit_transform(data_training)
x_train=[]
y_train=[]

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train, y_train =np.array(x_train),np.array(y_train)

# load my model-------------------------------------

import tensorflow as tf

# Define your model using TensorFlow's Keras API
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(units=120, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)


# ------------------------------------------------

# testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# final graph
st.subheader('Prediction vs Original')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time(in days)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_predicted)
st.write("Mean Squared Error on Testing Data:", mse_test)