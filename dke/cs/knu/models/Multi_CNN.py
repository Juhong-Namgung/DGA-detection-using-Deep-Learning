# Load Libraries

import warnings

import model_evaluate
import model_preproecess
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate, Conv2D
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam

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
            conv = MaxPooling1D(3)(conv)
            #conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            conv = Dropout(0.5)(conv)

            return conv

        # Multiple Conv Layers
        Conv2D

        conv1 = get_conv_layer(emb, kernel_size=5, filters=256)
        conv2 = get_conv_layer(conv1, kernel_size=4, filters=256)
        conv3 = get_conv_layer(conv2, kernel_size=2, filters=256)
        conv3 = Lambda(sum_1d, output_shape=(256,))(conv3)

        #merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(1024)(conv3)
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

with tf.device("/GPU:0"):
    epochs = 10
    batch_size = 64

    # Load data using model preprocessor
    preprocess = model_preproecess.Preprocessor()

    X_train, X_test, y_train, y_test = preprocess.load_data()

    # Define CNN model
    model_name = "MULTI_1DCNN"
    model = conv_fully()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

    y_pred = model.predict(X_test, batch_size=64)

    evaluator = model_evaluate.Evaluator()

    # Validation curves
    evaluator.plot_validation_curves(model_name, history)
    evaluator.print_validation_report(history)

    # Experimental result
    evaluator.calculate_measrue(model, X_test, y_test)

    # model.summary()

    # Save confusion matrix
    evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # Save final training model
    # save_model("../models/" + model_name + ".json", "../models/" + model_name + ".h5")
