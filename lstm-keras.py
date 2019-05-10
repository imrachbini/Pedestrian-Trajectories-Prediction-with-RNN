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
from keras.utils.training_utils import multi_gpu_model

load_weight = True
num_feature = 10
batch_size = 16
rnn_size = 512
epochs = 100
output_size = 1
learning_rate = 0.0005

training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('data_train.csv', num_feature)

X_train = np.concatenate((training_X, dev_X))
Y_train = np.concatenate((training_Y, dev_Y))
X_val = testing_X
Y_val = testing_X

with tf.device("/cpu:0"):
    model = Sequential()
    model.add(
        LSTM(rnn_size, 
            input_shape=(training_X.shape[1], training_X.shape[2])
        )
    )
    model.add(Dense(2))
    if load_weight:
        model.load_weight('misc/keras_lstm.h5')

gpu_model = multi_gpu_model(model, gpus=2)
gpu_model.compile(loss='mse', optimizer='adam')

history = gpu_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, Y_val), verbose=1, shuffle=False)

model.save('misc/keras_lstm.h5')
