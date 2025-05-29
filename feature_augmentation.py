import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def gradient_covariates(X):
    """
    Computes gradients for each feature in X and flattens gradient axes.
    """
    gradients = [np.gradient(X[i]) for i in range(X.shape[0])]  # [(dy, dx), ...]
    flattened_gradients = [grad for gradient in gradients for grad in gradient]  # Flatten (dy, dx) for each feature
    return np.array(flattened_gradients)  # Shape: (num_features * 2, height, width)

def trend_surface(X, degree=2):
    """
    Generates polynomial features for trend surface fitting for a 2D input (height, width).
    """
    height, width = X.shape  # Extract the dimensions
    x_coords, y_coords = np.meshgrid(range(width), range(height))  # Create coordinate grids
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)  # Flatten coordinates
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    trend = poly.fit_transform(coords)  # Fit polynomial features
    return trend.reshape(height, width, -1).transpose(2, 0, 1) 