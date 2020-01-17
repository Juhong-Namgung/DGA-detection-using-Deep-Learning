# Load Libraries
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import model_preproecess
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, concatenate
from keras.layers import Input, ELU, Embedding, BatchNormalization
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

with tf.device("/GPU:0"):
    def conv_fully(max_len=77, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):

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

        hidden2 = Dense(256)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization(mode=0)(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        hidden3 = Dense(64)(hidden2)
        hidden3 = ELU()(hidden3)
        hidden3 = BatchNormalization(mode=0)(hidden3)
        hidden3 = Dropout(0.5)(hidden3)

        # Output layer (last fully connected layer)
        # 마지막 클래스 결정하는 layer
        output = Dense(21, activation='softmax', name='main_output')(hidden3)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.precision, preprocess.recall, preprocess.fmeasure])
        return model

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

    def simple_lstm(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim,
                        input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # LSTM layer
        lstm = LSTM(lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)

        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(lstm)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.fmeasure, preprocess.recall, preprocess.precision])

        return model

    def bidirection_lstm_with_attention(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        # Bi-directional LSTM layer
        lstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)
        lstm = Dropout(0.2)(lstm)

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

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer_for_cnnbi(emb, kernel_size=5, filters=256):
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

        conv2 = get_conv_layer_for_cnnbi(emb, kernel_size=2, filters=256)
        conv3 = get_conv_layer_for_cnnbi(emb, kernel_size=3, filters=256)
        conv4 = get_conv_layer_for_cnnbi(emb, kernel_size=4, filters=256)
        conv5 = get_conv_layer_for_cnnbi(emb, kernel_size=5, filters=256)

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

# save validation curves(.png format)
def plot_validation_curves(historys, names):
    losses = []
    for history in historys:
        losses.append(history.history['loss'])

    epochs = range(1, 11)
    i = 0
    for name in names:
        plt.plot(epochs, losses[i], label=name)
        i = i + 1

    plt.xlabel('Epochs')
    plt.grid()
    plt.legend(loc='lower right')
    #plt.show()
    now = datetime.now()
    nowDatetime = now.strftime('%Y_%m_%d-%H%M%S')

    plt.savefig('./result/' + 'all_val_curve_' + nowDatetime + '.png')

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 64

    # Load data using model preprocessor
    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # define all models
    models_name = ["CNN", "LSTM", "BILSTM", "LSTM_ATT", "BILSTM_ATT", "CNN_BILSTM_ATT"]
    #models_name = ["CNN", "BILSTM", "LSTM_ATT"]
    models = [conv_fully(), simple_lstm(), bidirectional_lstm(), lstm_with_attention(), bidirection_lstm_with_attention(), cnn_bidirection_lstm_with_attention()]
    historys = []
    for model in models:
        historys.append(model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11))
        print('===========================================================================================================================================================================')
        print('===========================================================================================================================================================================')
        print()
        print()
    plot_validation_curves(historys, models_name)

    

