import numpy as np
import read_data
import pickle

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM


num_feature = 5
batch_size = 30
rnn_size = 400
output_size = 1
learning_rate = 0.0005

training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('data_train.csv', num_feature)

training_X = np.array(training_X)
training_Y = np.array(training_Y)
dev_X = np.array(dev_X)
dev_Y = np.array(dev_Y)
testing_X = np.array(testing_X)
testing_Y = np.array(testing_Y)

model = Sequential()
model.add(
    LSTM(rnn_size, 
         input_shape=(training_X.shape[1], training_X.shape[2])
    )
)
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

history = model.fit(training_X, training_Y, epochs=100, batch_size=num_feature,
                    validation_data=(dev_X, dev_Y), verbose=2, shuffle=False)

model.save('keras_lstm.h5')
