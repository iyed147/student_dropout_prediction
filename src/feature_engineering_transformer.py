# src/feature_engineering_transformer.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.feature_engineering import AdvancedFeatureEngineer


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    REQUIRED_COLUMNS = {
        'absences': 0,
        'failures': 0,
        'studytime': 2,
        'goout': 3,
        'G1': 10,
        'G2': 10,
        'famrel': 3,
        'famsup': 'no',
        'schoolsup': 'no',
        'address': 'U',
        'Dalc': 1,
        'Walc': 1,
        'health': 3,
        'freetime': 3,
        'age': 16
    }

    def __init__(self):
        self.engineer = AdvancedFeatureEngineer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineeringTransformer attend un DataFrame")

        df = X.copy()

        for col, default in self.REQUIRED_COLUMNS.items():
            if col not in df.columns:
                df[col] = default

        return self.engineer.create_all_features(df)
