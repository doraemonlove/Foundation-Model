import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true)) * 100


def SMAPE(pred, true):
    return np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100


def MASE(pred, true, m=1):
    numerator = np.mean(np.abs(pred - true))
    
    denominator = np.mean(np.abs(true[m:] - true[:-m]))
    
    return numerator / (denominator + 1e-8) 


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    smape = SMAPE(pred, true)
    mape = MAPE(pred, true)
    mase = MASE(pred, true)

    return mae, mse, smape, mape, mase
