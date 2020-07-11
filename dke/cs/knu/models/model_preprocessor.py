import pandas as pd
import numpy as np
from sklearn import model_selection
import tensorflow as tf

class Preprocessor:
    def __init__(self):
        pass

    def load_data(path="../data/dga_label.csv"):

        df = pd.read_csv(path, encoding='ISO-8859-1', sep=',')

        # Tokenizing domain string on character level
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=True, char_level=True)
        tokenizer.fit_on_texts(df['domain'])
        domain_tokens = tokenizer.texts_to_sequences(df['domain'])

        # Padding domain token max_len=74
        domain_max_len = 74

        x = tf.keras.preprocessing.sequence.pad_sequences(domain_tokens, maxlen=domain_max_len, padding="post")
        y = np.array(df['class'])

        # Cross validation
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

        # DGA class: 0~21
        y_train_category = tf.keras.utils.to_categorical(y_train, 21)
        y_test_category = tf.keras.utils.to_categorical(y_test, 21)

        return x_train, x_test, y_train_category, y_test_category