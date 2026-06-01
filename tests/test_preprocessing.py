# tests/test_preprocessing.py
import unittest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

class TestPreprocessing(unittest.TestCase):
    def test_preprocessor_creation(self):
        """Test création préprocesseur"""
        preprocessor = DataPreprocessor()
        self.assertIsNotNone(preprocessor)
    
    def test_fit_transform(self):
        """Test fit_transform"""
        df = pd.DataFrame({
            'age': [18, 19, 20],
            'absences': [0, 5, 10],
            'school': ['GP', 'MS', 'GP']
        })
        
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(df)
        
        self.assertEqual(X_processed.shape[0], 3)