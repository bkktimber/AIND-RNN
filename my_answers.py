import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    
    # containers for input/output pairs
    X = []
    for item in range(len(series) - window_size):
        X.append(series[item:item + window_size])

    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    assert(type(X).__name__ == 'ndarray')
    assert(type(y).__name__ == 'ndarray')

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    # Initialize the model
    model = Sequential()
    lstm_units = 5
    dense_units = 1

    # Layer 1: LSTM layer with 5 hidden units
    model.add(LSTM(lstm_units, activation='tanh', recurrent_activation='hard_sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True,
            input_shape= (window_size, 1)))
    
    # Layer 2: Fully Connected layer with 1 unit
    model.add(Dense(dense_units, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None))
    
    return model
    # pass


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # punctuation = ['!',',', '.',':', ';', '?']
    legit_char = str('abcdefghijklmnopqrstuvwxyz!,.:;? ')
    filtered = ''.join(filter(lambda x: x in legit_char, text))
    return filtered

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    for item in range(0, len(text) - window_size, step_size):
        inputs.append(text[item:item + window_size])

    outputs = []
    for alphabet in range(window_size, len(text), step_size):
        outputs.append(text[alphabet])

    assert(type(inputs).__name__ == 'list')
    assert(type(outputs).__name__ == 'list')

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # Initilize model
    model = Sequential()
    lstm_units = 200
    dense_units = num_chars

    # Layer 1: LSTM layer with 200 hidden units
    model.add(LSTM(lstm_units, activation='tanh', recurrent_activation='hard_sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True,
            input_shape= (window_size, num_chars)))

    # Layer 2: Fully Connected layer with no. of hidden unit equally to no. unique characters in dataset
    model.add(Dense(dense_units, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None))

    # Layer 3: Softmax Activation layer
    model.add(Activation('softmax'))
    return model
    # pass
