"""
Tests unitaires pour le module de modélisation
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Ajouter le chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.modeling import RobustModelPipeline


class TestModeling(unittest.TestCase):
    """Tests pour le module de modélisation"""

    def setUp(self):
        np.random.seed(42)

        self.X_train = pd.DataFrame({
            'age': np.random.randint(15, 22, 100),
            'absences': np.random.randint(0, 30, 100),
            'G1': np.random.randint(0, 21, 100),
            'G2': np.random.randint(0, 21, 100),
            'failures': np.random.randint(0, 3, 100),
            'studytime': np.random.randint(1, 5, 100),
            'school': np.random.choice(['GP', 'MS'], 100),
            'sex': np.random.choice(['F', 'M'], 100),
            'famsup': np.random.choice(['yes', 'no'], 100)
        })

        self.y_train = np.array([0] * 80 + [1] * 20)
        np.random.shuffle(self.y_train)

        self.X_test = self.X_train.iloc[:20]
        self.y_test = self.y_train[:20]

        self.preprocessor = Mock()
        self.preprocessor.fit_transform.return_value = np.random.randn(100, 10)
        self.preprocessor.transform.return_value = np.random.randn(20, 10)

    def test_pipeline_initialization(self):
        pipeline = RobustModelPipeline()
        self.assertEqual(pipeline.models, {})
        self.assertIsNone(pipeline.best_model)

    def test_train_models_method_exists(self):
        pipeline = RobustModelPipeline()
        self.assertTrue(hasattr(pipeline, 'train_models'))

    @patch('src.modeling.GridSearchCV')
    @patch('src.modeling.RandomForestClassifier')
    def test_train_models_with_mocks(self, mock_rf, mock_grid):
        mock_grid_instance = Mock()
        mock_grid_instance.best_estimator_ = Mock()
        mock_grid_instance.best_estimator_.predict.return_value = np.array([0, 1])
        mock_grid_instance.best_estimator_.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
        mock_grid_instance.best_score_ = 0.85
        mock_grid_instance.best_params_ = {'n_estimators': 100}
        mock_grid.return_value = mock_grid_instance

        pipeline = RobustModelPipeline()

        results = pipeline.train_models(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.preprocessor
        )

        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)

    def test_model_selection_logic(self):
        results_df = pd.DataFrame({
            'Model': ['RF', 'XGB', 'LR'],
            'Test_Recall': [0.85, 0.87, 0.82],
            'False_Negatives': [3, 2, 4]
        })

        best_model = results_df.loc[
            results_df['Test_Recall'].idxmax(), 'Model'
        ]

        self.assertEqual(best_model, 'XGB')

    def test_class_imbalance(self):
        unique, counts = np.unique(self.y_train, return_counts=True)
        ratio = counts[0] / counts[1]
        self.assertEqual(ratio, 4.0)

    def test_metrics_computation(self):
        from sklearn.metrics import confusion_matrix

        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        self.assertEqual(tp, 2)
        self.assertEqual(fn, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(tn, 2)


class TestSMOTEIntegration(unittest.TestCase):
    """Tests légers autour de SMOTE"""

    def test_imblearn_available(self):
        try:
            from imblearn.over_sampling import SMOTE
            self.assertTrue(True)
        except ImportError:
            self.fail("imblearn n'est pas installé")


def run_tests():
    print("🔧 Lancement des tests modeling...")
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModeling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSMOTEIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
