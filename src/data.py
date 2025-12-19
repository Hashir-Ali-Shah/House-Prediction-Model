
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_data(path="Data/housing.csv"):
    return pd.read_csv(path)

# Indices for custom transformer
# 'total_rooms', 'total_bedrooms', 'population', 'households'
# Based on housing.columns: 
# ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
# Indices: 0, 1, 2, 3, 4, 5, 6, 7
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def get_data_pipeline(housing):
    """
    Creates and returns the full data processing pipeline.
    """
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    
    return full_pipeline

def preprocess_data(housing):
    """
    Preprocesses the raw dataframe.
    Returns:
        X_prepared: processed features
        y: target labels (median_house_value)
        pipeline: fitted pipeline object
    """
    # Create StratifiedShuffleSplit equivalent logic if needed, 
    # but for simplicity we will just train on the whole set or similar 
    # to the notebook's simple fit_transform on full data for now.
    
    # In Data.ipynb they separate X and y
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)
    
    pipeline = get_data_pipeline(housing_features)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    return housing_prepared, housing_labels, pipeline
