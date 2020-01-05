import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np

class Evaluator:
    def __init__(self):
        self.id = 1

    def plot_validation_curves(self, history):
        history_dict = history.history
        print(history_dict.keys())

        # validation curves
        epochs = range(1, len(history_dict['loss']) + 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, history_dict['val_fmeasure'], 'r',label='f1')
        plt.plot(epochs, history_dict['val_precision'], 'g',label='precision')
        plt.plot(epochs, history_dict['val_recall'], 'k',label='recall')
        plt.plot(epochs, history_dict['val_categorical_accuracy'], 'c', label='categorical_accuracy')

        plt.xlabel('Epochs')
        plt.grid()
        plt.legend(loc=1)
        plt.show()

    def calculate_measrue(self, model, X_test, y_test):
        # Calculate measure(categorical accuracy, precision, recall, f1-score)
        y_pred_class_prob = model.predict(X_test, batch_size=64)
        y_pred_class = np.argmax(y_pred_class_prob, axis=1)
        y_test_class = np.argmax(y_test, axis=1)
        y_val_class = y_test_class

        # classification report(sklearn)
        print(classification_report(y_val_class, y_pred_class, digits=4))

        print ("precision" , metrics.precision_score(y_val_class, y_pred_class, average = 'weighted'))
        print ("recall" , metrics.recall_score(y_val_class, y_pred_class, average = 'weighted'))
        print ("f1" , metrics.f1_score(y_val_class, y_pred_class, average = 'weighted'))





















