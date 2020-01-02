# Load Libraries
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.layers.core import Dense, Dropout
from keras.layers import Input, LSTM, Embedding
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from . import model_preproecess

import warnings
warnings.filterwarnings("ignore")

with tf.device("/GPU:0"):

    def simple_lstm(max_len=74, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim,
                        input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # LSTM layer
        lstm = LSTM(lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)

        # Output layer (last fully connected layer)
        output = Dense(20, activation='sigmoid', name='output')(lstm)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', preprocess.fmeasure, preprocess.recall, preprocess.precision])

        return model

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 64

    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    model = simple_lstm()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    history_dict = history.history
    print(history_dict.keys())
    epochs = range(1, len(history_dict['loss']) + 1)

    # "bo" is for "blue dot"
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

    print(classification_report(y_val_class, y_pred_class))

    # Save final training model
    # model_name = "LSTM"
    # preprocess.save_model(model, "../models/" + model_name + ".json", "../models/" + model_name + ".h5")
