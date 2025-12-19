
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict

def train_linear_regression(X, y):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_decision_tree(X, y):
    """Trains a Decision Tree Regressor."""
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model

def train_random_forest(X, y, n_estimators=100, max_features=1.0):
    """Trains a Random Forest Regressor."""
    # max_features='auto' is deprecated in 1.2+
    model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Evaluates model using RMSE on the given set (typically training set if no test set provided).
    Slightly biased but consistent with notebook's basic checks.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    return rmse

def cross_validate_model(model, X, y, cv=10):
    """Performs cross-validation and returns RMSE scores."""
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

def get_cv_predictions(model, X, y, cv=10):
    """Generates cross-validated predictions for more realistic visualization."""
    predictions = cross_val_predict(model, X, y, cv=cv)
    return predictions
