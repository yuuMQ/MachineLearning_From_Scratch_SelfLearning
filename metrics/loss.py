import numpy as np

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def r2_score(y_true, y_pred):
    # SSres
    SSres = np.sum((y_true - y_pred)**2)
    # SStot
    y_mean = np.mean(y_true)
    SStot = np.sum((y_true - y_mean)**2)
    # R2
    r2 = 1 - SSres / SStot
    return r2

if __name__ == '__main__':
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'mse: {mse}')
    print(f'mae: {mae}')
    print(f'r2: {r2}')

