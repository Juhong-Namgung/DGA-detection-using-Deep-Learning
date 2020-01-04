# Load Libraries
import warnings

# from . import model_preproecess
import model_evaluate
import model_preproecess
import tensorflow as tf
from keras import regularizers
from keras.layers import Bidirectional
from keras.layers import Input, LSTM, Embedding
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

warnings.filterwarnings("ignore")

with tf.device("/GPU:0"):

    def bidirectional_lstm(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim,
                        input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # LSTM layer
        lstm = Bidirectional(LSTM(units=128, recurrent_dropout=0.2))(emb)
        lstm = Dropout(0.5)(lstm)

        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(lstm)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.precision, preprocess.recall, preprocess.fmeasure])

        return model

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 64


    # Load data using model preprocessor
    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # define BILSTM model
    model = bidirectional_lstm()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

    evaluator = model_evaluate.Evaluator()

    # validation curves
    #evaluator.plot_validation_curves(history)

    # experimental result
    evaluator.calculate_measrue(model, X_test, y_test)

    # Save final training model
    # model_name = "BILSTM"
    # preprocess.save_model(model, "../models/" + model_name + ".json", "../models/" + model_name + ".h5")
