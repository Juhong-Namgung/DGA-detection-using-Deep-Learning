# Load Libraries
import warnings

import model_evaluate
import model_preproecess
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Input, ELU, Embedding, BatchNormalization
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):

    def lstm_with_attention(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # LSTM layer
        lstm = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(emb)
        lstm = Dropout(0.2)(lstm)

        # Attention layer
        att = SeqSelfAttention(attention_activation='relu')(lstm)
        att = Flatten()(att)

        hidden1 = Dense(4928)(att)
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
    epochs = 10
    batch_size = 64

    # Load data using model preprocessor
    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # define LSTM with attention model
    model_name = "LSTM_ATT"
    model = lstm_with_attention()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

    y_pred = model.predict(X_test, batch_size=64)

    evaluator = model_evaluate.Evaluator()

    # Validation curves
    evaluator.plot_validation_curves(model_name, history)
    evaluator.print_validation_report(history)

    # Experimental result
    evaluator.calculate_measrue(model, X_test, y_test)

    # Save confusion matrix
    evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # model.summary()

    # Save final training model
    # preprocess.save_model(model, "../models/" + model_name + ".json", "../models/" + model_name + ".h5")
