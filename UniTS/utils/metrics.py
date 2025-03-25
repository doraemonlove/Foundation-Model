import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 *
                (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    # 避免除以零
    mask = true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

def SMAPE(pred, true):
    # 避免除以零
    mask = true != 0
    if not np.any(mask):
        return np.nan
    return 200 * np.mean(np.abs(pred[mask] - true[mask]) / (np.abs(pred[mask]) + np.abs(true[mask])))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mape = MAPE(pred, true)
    smape = SMAPE(pred, true)

    return mae, mse, mape, smape
