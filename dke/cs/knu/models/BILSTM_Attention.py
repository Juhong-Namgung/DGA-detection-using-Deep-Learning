# Load Libraries
import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import KFold
from pathlib import Path
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python.platform import gfile
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras_self_attention import SeqSelfAttention

import json

import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

# General save model to disk function
def save_model(fileModelJSON, fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)


with tf.device("/GPU:0"):

    # Load data
    DATA_HOME ='../data/'
    df = pd.read_csv(DATA_HOME + 'dga_label_shuffle.csv',encoding='ISO-8859-1', sep=',')

    # Convert domain string to integer
    # URL 알파벳을 숫자로 변경
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.domain]

    # Padding domain integer max_len=74
    # 최대길이 74로 지정
    max_len = 74

    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    y = np.array(df['class'])

    # Cross-validation
    X_train, X_test, y_train0, y_test0 = model_selection.train_test_split(X, y, test_size=0.2, random_state=33)

    # dga class: 0~20: 21개
    y_train = np_utils.to_categorical(y_train0, 21)
    y_test = np_utils.to_categorical(y_test0, 21)

with tf.device("/GPU:0"):

    def bidirection_lstm_with_attention(max_len=74, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input) 

        # Bi-directional LSTM layer
        lstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)
        lstm = Dropout(0.2)(lstm)
        
        att = SeqSelfAttention(attention_activation='relu')(lstm)
        att = Flatten()(att)
    
        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(att)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

with tf.device("/GPU:0"):
    epochs = 4
    batch_size = 64

    model = bidirection_lstm_with_attention()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('\nFinal Cross-Validation Accuracy', accuracy, '\n')

    # Save final training model
    # model_name = "BILSTM"
    # save_model("../models/" + model_name + ".json", "../models/" + model_name + ".h5")
