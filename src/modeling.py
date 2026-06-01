import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, fbeta_score, recall_score, precision_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class RobustModelPipeline:
    """
    Pipeline de modélisation avec SMOTE appliqué DANS chaque fold de CV.
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}

    # ------------------------------------------------------------------
    def _make_pipeline(self, model, preprocessor):
        """Pipeline : prétraitement → SMOTE → modèle."""
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("sampler", SMOTE(random_state=42)),
            ("classifier", model),
        ])

    # ------------------------------------------------------------------
    def train_models(self, X_train, y_train, X_test, y_test, preprocessor):
        print("\n🤖 MODÉLISATION AVEC SMOTE CORRECT")
        print("=" * 50)

        model_configs = {
            "Random Forest": {
                "model": RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
                "params": {
                    "classifier__n_estimators": [80, 100],
                    "classifier__max_depth": [3, 4],
                    "classifier__min_samples_split": [25, 30],
                    "classifier__min_samples_leaf": [10, 15],
                    "classifier__max_features": [0.5],
                    "classifier__max_leaf_nodes": [10, 15],
                    "sampler__k_neighbors": [3],
                },
            },
            "XGBoost": {
                "model": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                    max_depth=2,
                    min_child_weight=15,
                    gamma=1.5,
                    subsample=0.55,
                    colsample_bytree=0.55,
                    reg_alpha=2.0,
                    reg_lambda=3.0,
                ),
                "params": {
                    "classifier__n_estimators": [30, 40],
                    "classifier__learning_rate": [0.05],
                    "sampler__k_neighbors": [3],
                },
            },
            "Logistic Regression": {
                "model": LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=200,
                    solver="liblinear",
                ),
                "params": {
                    "classifier__C": [0.1, 0.5, 1],
                    "sampler__k_neighbors": [3],
                },
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(n_jobs=-1),
                "params": {
                    "classifier__n_neighbors": [7, 9],
                    "classifier__weights": ["distance"],
                    "classifier__metric": ["manhattan"],
                    "sampler__k_neighbors": [3],
                },
            },
        }

        f2_scorer = make_scorer(fbeta_score, beta=2)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f2": f2_scorer,
            "roc_auc": "roc_auc",
        }

        results_list = []

        for name, config in model_configs.items():
            print(f"\n📊 Entraînement: {name}")
            print("-" * 30)

            pipeline = self._make_pipeline(config["model"], preprocessor)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["params"],
                cv=cv,
                scoring=scoring,
                refit="f2",
                n_jobs=-1,
                verbose=0,
                return_train_score=True,
                error_score="raise",
            )

            grid_search.fit(X_train, y_train)
            self.models[name] = grid_search.best_estimator_

            y_pred = grid_search.best_estimator_.predict(X_test)
            y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            metrics = {
                "Model": name,
                "Best_CV_F2_Score": grid_search.best_score_,
                "Test_Accuracy": accuracy_score(y_test, y_pred),
                "Test_Precision": precision_score(y_test, y_pred, zero_division=0),
                "Test_Recall": recall_score(y_test, y_pred),
                "Test_F1_Score": f1_score(y_test, y_pred),
                "Test_F2_Score": fbeta_score(y_test, y_pred, beta=2),
                "Test_ROC_AUC": roc_auc_score(y_test, y_pred_proba),
                "True_Positives": tp,
                "False_Negatives": fn,
                "False_Positives": fp,
                "True_Negatives": tn,
                "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "Best_Params": str(grid_search.best_params_),
            }

            results_list.append(metrics)
            self.cv_results[name] = grid_search.cv_results_

            print(f"  F2-Score (CV)  : {grid_search.best_score_:.3f}")
            print(f"  Recall (Test)  : {metrics['Test_Recall']:.3f}")
            print(f"  Précision      : {metrics['Test_Precision']:.3f}")
            print(f"  Faux Négatifs  : {fn}")
            print(f"  Faux Positifs  : {fp}")

        results_df = pd.DataFrame(results_list)

        results_df["Education_Score"] = (
            0.5 * results_df["Test_Recall"]
            + 0.2 * results_df["Test_F1_Score"]
            + 0.2 * results_df["Test_ROC_AUC"]
            + 0.1 * (1 - results_df["False_Positives"] / len(y_test))
        )

        print("\n" + "=" * 60)
        print("🏆 COMPARAISON DES MODÈLES")
        print("=" * 60)
        cols = ["Model", "Test_Recall", "Test_Precision",
                "Test_F2_Score", "Education_Score",
                "False_Negatives", "False_Positives"]
        print(
            results_df[cols]
            .sort_values("Education_Score", ascending=False)
            .to_string(index=False)
        )

        best_idx = results_df["Education_Score"].idxmax()
        self.best_model_name = results_df.loc[best_idx, "Model"]
        self.best_model = self.models[self.best_model_name]

        print(f"\n✅ MEILLEUR MODÈLE: {self.best_model_name}")
        print(f"   Education Score : {results_df.loc[best_idx, 'Education_Score']:.3f}")
        print(f"   Recall          : {results_df.loc[best_idx, 'Test_Recall']:.3f}")
        print(f"   F2-Score        : {results_df.loc[best_idx, 'Test_F2_Score']:.3f}")
        print(f"   Faux Négatifs   : {int(results_df.loc[best_idx, 'False_Negatives'])}")

        return results_df

    # ------------------------------------------------------------------
    def save_best_model(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "best_model.pkl")
        joblib.dump(self.best_model, model_path)

        for name, model in self.models.items():
            safe = name.lower().replace(" ", "_")
            joblib.dump(model, os.path.join(path, f"model_{safe}.pkl"))

        print(f"\n💾 Modèles sauvegardés dans: {path}")
        print(f"   - best_model.pkl ({self.best_model_name})")
        return model_path

    def load_best_model(self, path="models/best_model.pkl"):
        self.best_model = joblib.load(path)
        print(f"✅ Modèle chargé: {path}")
        return self.best_model