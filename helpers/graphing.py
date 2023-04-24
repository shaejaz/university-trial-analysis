import numpy as np
import matplotlib.pyplot as plt

def plot_keras_model_logistic_regression_accuracy(model_history, logsitic_regression_score, legend_names, filename='basic_lstm_log_reg.png'):
    """
    Plots the accuracy of the Keras model and the logistic regression model
    """
    plt.plot(100*np.asarray(model_history.history['accuracy']))
    plt.plot(100*np.asarray(model_history.history['val_accuracy']), color='g')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 100)

    plt.axhline(y=logsitic_regression_score*100, color='r')

    plt.legend(legend_names, loc='lower left')

    plt.savefig(filename)
    plt.show()