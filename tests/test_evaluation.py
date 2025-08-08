from math import sqrt
from sklearn.metrics import mean_squared_error

def test_rmse_calculation():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    assert round(rmse, 2) == 0.71
