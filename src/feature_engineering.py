# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class AdvancedFeatureEngineer:
    """Classe pour le feature engineering avancé"""

    def __init__(self):
        self.created_features = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n🔧 FEATURE ENGINEERING AVANCÉ")
        print("=" * 50)

        df = df.copy()
        original_shape = df.shape

        df = self.create_interaction_features(df)
        df = self.create_polynomial_features(df, degree=2)
        df = self.create_composite_scores(df)
        df = self.create_temporal_features(df)
        df = self.create_behavioral_features(df)

        print("\n✅ Feature engineering terminé:")
        print(f"   Avant: {original_shape[1]} colonnes")
        print(f"   Après: {df.shape[1]} colonnes")
        print(f"   {len(self.created_features)} nouvelles features créées")

        return df

    def create_interaction_features(self, df):
        df['absences_per_failure'] = df['absences'] / (df['failures'] + 1.0)
        df['study_vs_social'] = df['studytime'] / (df['goout'] + 1.0)
        df['total_family_support'] = (
            (df['famsup'] == 'yes').astype(int) +
            (df['famrel'] / 5.0)
        )
        df['efficiency_score'] = (
            ((df['G1'] + df['G2']) / 2.0) /
            (df['studytime'] + 1.0)
        )
        return df

    def create_polynomial_features(self, df, degree=2):
        numeric_features = ['absences', 'G1', 'G2', 'failures', 'studytime', 'age']
        existing = [f for f in numeric_features if f in df.columns]

        if len(existing) >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            values = poly.fit_transform(df[existing])
            names = poly.get_feature_names_out(existing)

            for i, name in enumerate(names):
                if name not in df.columns:
                    df[name] = values[:, i]

        return df

    def create_composite_scores(self, df):
        df['academic_risk_score'] = (
            (df['failures'] * 2.5) +
            ((df['absences'] > 10).astype(int) * 2.0) +
            ((df['studytime'] < 2).astype(int) * 1.5)
        )
        return df

    def create_temporal_features(self, df):
        df['grade_trend'] = df['G2'] - df['G1']
        return df

    def create_behavioral_features(self, df):
        df['life_balance'] = (df['studytime'] / 4.0) - (df['goout'] / 5.0)
        return df
