from sklearn.linear_model import LinearRegression
import numpy as np
from math import atan

def linear_regression_slope_scale(data, begin, end, angle=True):
    X = np.arange(1, 5).reshape(-1, 1) 
    y = data[begin:end]
    model_term = LinearRegression().fit(X, y)
    slope_term = model_term.coef_[0] 
    if angle:
        return slope_term 
    return atan(slope_term)

def linear_regression_slope(data, begin, end, angle=True):
    X = np.arange(1, 5).reshape(-1, 1) 
    y = data[begin:end]
    model_term = LinearRegression().fit(X, y)
    slope_term = model_term.coef_[0] 
    return slope_term


def two_point_slope_Scale(data, begin, end, angle=True):
    m = (data[end] - data[begin]) / (end - begin)

    if angle: 
        return atan(m)
    return m

def two_point_slope(data, begin, end, angle=True):
    m = (data[end] - data[begin]) / (end - begin)

    return m