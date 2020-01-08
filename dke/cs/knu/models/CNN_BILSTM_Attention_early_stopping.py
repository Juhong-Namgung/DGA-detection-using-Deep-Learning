# Load Libraries
import warnings

import model_evaluate
import model_preproecess
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.core import Flatten
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

warnings.filterwarnings("ignore")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
        conv = ELU()(conv)
        #conv = MaxPooling1D(5)(conv)
        #conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)
        return conv

    def cnn_bidirection_lstm_with_attention(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        conv2 = get_conv_layer(emb, kernel_size=2, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv4 = get_conv_layer(emb, kernel_size=4, filters=256)
        conv5 = get_conv_layer(emb, kernel_size=5, filters=256)

        merged = concatenate([conv2, conv3, conv4, conv5])

        # Bi-directional LSTM layer
        lstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)

        att = SeqSelfAttention(attention_activation='relu')(lstm)

        cnnlstm_merged = concatenate([merged, att])
        cnnlstm_merged = Flatten()(cnnlstm_merged)

        hidden1 = Dense(640)(cnnlstm_merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(hidden1)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.fmeasure, preprocess.recall, preprocess.precision])
        return model

with tf.device("/GPU:0"):
    epochs = 30
    batch_size = 64

    # Load data using model preprocessor
    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # Define LSTM with attention model
    model_name = "CNN_BILSTM_ATT_early"
    model = cnn_bidirection_lstm_with_attention()

    # Define early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('./trained_model/' + model_name+ '.h5', monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[es, mc])

    saved_model = load_model('./trained_model/' + model_name+ '.h5', compile=False, custom_objects={'SeqSelfAttention':SeqSelfAttention})
    y_pred = saved_model.predict(X_test, batch_size=64)

    evaluator = model_evaluate.Evaluator()

    # Validation curves
    evaluator.plot_validation_curves(model_name, history)
    evaluator.print_validation_report(history)

    # Experimental result
    evaluator.calculate_measrue(saved_model, X_test, y_test)

    # Save confusion matrix
    evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # Save final training model
    # preprocess.save_model(model, "../models/" + model_name + ".json", "../models/" + model_name + ".h5")

    #model.summary()