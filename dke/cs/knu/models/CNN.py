import warnings

import tensorflow as tf
import tensorflow_addons as tfa
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import layers

# GPU memory setting
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

warnings.filterwarnings('ignore')
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def conv_fully(max_len=74, emb_dim=32, max_vocab_len=40, W_reg=tf.keras.regularizers.l2(1e-4)):
    """CNN model with the Keras functional API"""

    # Input
    main_input = keras.Input(shape=(max_len,), dtype='int32', name='main_input')

    # Embedding layer
    emb = layers.Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, embeddings_regularizer=W_reg)(main_input)
    emb = layers.Dropout(0.2)(emb)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = layers.Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
        conv = layers.ELU()(conv)
        conv = layers.MaxPooling1D(5)(conv)
        conv = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), output_shape=(filters,))(conv)
        conv = layers.Dropout(0.5)(conv)

        return conv

    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

    merged = layers.concatenate([conv1, conv2, conv3, conv4], axis=1)

    hidden1 = layers.Dense(1024, activation='relu')(merged)
    hidden1 = layers.ELU()(hidden1)
    hidden1 = layers.BatchNormalization()(hidden1)
    hidden1 = layers.Dropout(0.5)(hidden1)

    hidden2 = layers.Dense(1024, activation='relu')(hidden1)
    hidden2 = layers.ELU()(hidden2)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(0.5)(hidden2)

    main_output = layers.Dense(21, activation='softmax')(hidden2)

    cnn_model = tf.keras.Model(inputs=main_input, outputs=main_output)

    return cnn_model


x_train, x_test, y_train, y_test = Preprocessor.load_data()

model_name = "CNN"
model = conv_fully()
model.summary()

# Plot model(.png)
# tf.keras.utils.plot_model(model, to_file="./result/" + model_name)

epochs = 10
batch_size = 64
adam = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
checkpoint_filepath = './tmp/checkpoint/' +  model_name + '.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                                               save_best_only=True, mode='auto')

model.compile(optimizer=adam, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                       tfa.metrics.F1Score(num_classes=21, average='weighted', name="f1_score_weighted"),
                       tfa.metrics.F1Score(num_classes=21, average='micro', name="f1_score_micro"),
                       tfa.metrics.F1Score(num_classes=21, average='macro', name="f1_score_macro")])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)   # Error in Windows environment

# Validation curves
Evaluator.plot_validation_curves(model_name, history)

y_pred = model.predict(x_test, batch_size=64)

# Experiment result
Evaluator.calculate_measure(model, x_test, y_test)
Evaluator.plot_confusion_matrix(model_name, y_test, y_pred)
Evaluator.plot_roc_curves(model_name, y_test, y_pred)