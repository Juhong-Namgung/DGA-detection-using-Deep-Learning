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
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report

from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python.platform import gfile

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
    #df = pd.read_csv(DATA_HOME + 'sample.csv',encoding='ISO-8859-1', sep=',')

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

def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.
-    The F score is the weighted harmonic mean of precision and recall.
-    Here it is only computed as a batch-wise average, not globally.
-    This is useful for multi-label classification, where input samples can be
-    classified as sets of labels. By only using accuracy (precision) a model
-    would achieve a perfect score by simply assigning every class to every
-    input. In order to avoid this, a metric should penalize incorrect class
-    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
-    computes this, as a weighted mean of the proportion of correct class
-    assignments vs. the proportion of incorrect class assignments.
-    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
-    correct classes becomes more important, and with beta > 1 the metric is
-    instead weighted towards penalizing incorrect class assignments.
-    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

with tf.device("/GPU:0"):

    def conv_fully(max_len=74, emb_dim=32, max_vocab_len=100, W_reg=regularizers.l2(1e-4)):

        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        # URL을 int로 변환한 것을 임베딩
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(main_input)
        emb = Dropout(0.25)(emb)

        def sum_1d(X):
            return K.sum(X, axis=1)

        def get_conv_layer(emb, kernel_size=5, filters=256):
            # Conv layer
            conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
            conv = ELU()(conv)
            conv = MaxPooling1D(5)(conv)
            conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            conv = Dropout(0.5)(conv)

            return conv

        # Multiple Conv Layers
        # 커널 사이즈를 다르게 한 conv
        conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
        conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
        conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

        # Fully Connected Layers
        # 위 결과 합침
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(1024)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization(mode=0)(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        hidden3 = Dense(256)(hidden2)
        hidden3 = ELU()(hidden3)
        hidden3 = BatchNormalization(mode=0)(hidden3)
        hidden3 = Dropout(0.5)(hidden3)

        hidden4 = Dense(64)(hidden3)
        hidden4 = ELU()(hidden4)
        hidden4 = BatchNormalization(mode=0)(hidden4)
        hidden4 = Dropout(0.5)(hidden4)

        # Output layer (last fully connected layer)
        # 마지막 클래스 결정하는 layer
        output = Dense(21, activation='softmax', name='main_output')(hidden4)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', fmeasure, recall, precision])
        return model

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 16

    model = conv_fully()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    history_dict = history.history
    print(history_dict.keys())
    epochs = range(1, len(history_dict['loss']) + 1)

    # "bo" is for "blue dot"
    #plt.plot(epochs, history_dict['loss'], 'b',label='loss')
    plt.plot(epochs, history_dict['fmeasure'], 'r',label='f1')
    plt.plot(epochs, history_dict['precision'], 'g',label='precision')
    plt.plot(epochs, history_dict['recall'], 'k',label='recall')

    plt.xlabel('Epochs')
    plt.grid()
    plt.legend(loc=1)
    plt.show()

    y_pred_class_prob = model.predict(X_test, batch_size=64)
    y_pred_class = np.argmax(y_pred_class_prob, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
    y_val_class = y_test_class

    print ("precision" , metrics.precision_score(y_val_class, y_pred_class, average = 'weighted'))
    print ("recall" , metrics.recall_score(y_val_class, y_pred_class, average = 'weighted'))
    print ("f1" , metrics.f1_score(y_val_class, y_pred_class, average = 'weighted'))

    # loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    accuracy = model.evaluate(X_test, y_test, verbose=1)

    print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
    print(classification_report(y_val_class, y_pred_class))

    # # Save final training model
    # model_name = "1DCNN"
    # save_model("../models/" + model_name + ".json", "../models/" + model_name + ".h5")
    #model.summary()

