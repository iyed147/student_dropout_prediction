#!/usr/bin/env python3
"""
Script principal — pipeline complet de prédiction d'abandon étudiant.
À lancer depuis la RACINE du projet :
    python main.py
"""

import sys
import os

# --- PYTHONPATH : racine du projet en premier ---
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)


def main():
    print("🎓 PROJET: PRÉDICTION D'ABANDON ÉTUDIANT")
    print("=" * 60)

    # 1. Chargement des données
    print("\n📥 ÉTAPE 1: Chargement des données")
    from src.data_loader import StudentDataLoader
    loader = StudentDataLoader()
    data = loader.load_data()

    # 2. Variable cible
    print("\n🎯 ÉTAPE 2: Création de la variable cible")
    from src.target_creation import TargetCreator
    target_creator = TargetCreator()
    data = target_creator.create_dropout_target(data)

    # 3. Split train / test
    print("\n📊 ÉTAPE 3: Séparation train/test")
    from sklearn.model_selection import train_test_split
    X = data.drop("dropout_risk", axis=1)
    y = data["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train : {X_train.shape}")
    print(f"  Test  : {X_test.shape}")

    # 4. Prétraitement
    print("\n⚙️ ÉTAPE 4: Prétraitement")
    from src.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train, y_train)   # fit() suffit — fit_transform() jetterait X transformé

    # 5. Modélisation (SMOTE dans chaque fold de CV)
    print("\n🤖 ÉTAPE 5: Modélisation (SMOTE dans pipeline)")
    from src.modeling import RobustModelPipeline
    model_pipeline = RobustModelPipeline()
    results_df = model_pipeline.train_models(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # 6. Évaluation complète
    print("\n📈 ÉTAPE 6: Évaluation complète")
    from src.evaluation import ModelEvaluator
    evaluator = ModelEvaluator()
    best_model = model_pipeline.best_model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    evaluation_results = evaluator.comprehensive_evaluation(
        y_test, y_pred, y_pred_proba, model_pipeline.best_model_name
    )

    # 7. Importance des features
    print("\n🔍 ÉTAPE 7: Importance des features")
    if preprocessor.feature_names is not None:
        evaluator.plot_feature_importance(best_model, preprocessor.feature_names)

    # 8. Sauvegarde
    print("\n💾 ÉTAPE 8: Sauvegarde des artefacts")
    model_pipeline.save_best_model("models/")

    # Résumé
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    best_row = results_df[results_df["Model"] == model_pipeline.best_model_name]
    print(f"\n📊 RÉSUMÉ DES RÉSULTATS:")
    print(f"  Meilleur modèle : {model_pipeline.best_model_name}")
    print(f"  Recall (test)   : {best_row['Test_Recall'].values[0]:.3f}")
    print(f"  F2-Score (test) : {best_row['Test_F2_Score'].values[0]:.3f}")
    print(f"  Faux négatifs   : {int(best_row['False_Negatives'].values[0])}")

    return {
        "model": best_model,
        "results": results_df,
        "evaluation": evaluation_results,
    }


if __name__ == "__main__":
    main()