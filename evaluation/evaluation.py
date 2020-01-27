from sklearn.metrics import mean_absolute_error
import numpy as np

def MAEScore(y_true, y_pred, targetVar):

    dictionary = {}
    for signal in range(len(targetVar)):
        y_true_signal = y_true[:,signal]
        y_pred_signal = y_pred[:,signal]
        dictionary.update({targetVar[signal]: [mean_absolute_error(y_true_signal, y_pred_signal)]})
    return dictionary


EPSILON = 1e-10
def MAAPEScore(group, target):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    actual = group.actual
    predicted = group[target]
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))