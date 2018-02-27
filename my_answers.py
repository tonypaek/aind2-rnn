import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = series[[j for i in range(len(series)-window_size) for j in range(i,i+window_size)]]
    y = series[window_size:]    

    # reshape each 
    X.shape = (int(len(X)/window_size),window_size)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):  
    #punctuation = ['!', ',', '.', ':', ';', '?']
    text=re.sub('[^a-z\!\,\.\:\;\?]+', ' ', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    end=window_size
    while end<len(text):
        inputs.append(text[end-window_size:end])
        outputs.append(text[end])
        end+=step_size
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200,input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    #model.compile(loss='mean_squared_error', optimizer=sgd)
    return model