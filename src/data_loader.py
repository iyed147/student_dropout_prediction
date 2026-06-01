import pandas as pd
import numpy as np
import os


class StudentDataLoader:
    """Charge et prépare les datasets étudiants."""

    def __init__(self, data_path="data"):
        self.data_path = data_path

    def load_data(self):
        print("📥 Chargement des données étudiantes...")

        math_path = os.path.join(self.data_path, "student-mat.csv")
        por_path  = os.path.join(self.data_path, "student-por.csv")

        if not os.path.exists(math_path) or not os.path.exists(por_path):
            print("⚠️  Fichiers CSV non trouvés → données de démonstration")
            return self._create_demo_data()

        df_math = pd.read_csv(math_path, sep=";")
        df_por  = pd.read_csv(por_path,  sep=";")

        print(f"  Math       : {df_math.shape}")
        print(f"  Portugais  : {df_por.shape}")

        df_math["course"] = "math"
        df_por["course"]  = "portuguese"

        df = pd.concat([df_math, df_por], ignore_index=True)
        print(f"✅ Données combinées : {df.shape}")
        return df

    # ------------------------------------------------------------------
    def _create_demo_data(self, n_samples=650):
        print("🔄 Création de données de démonstration...")
        np.random.seed(42)

        data = {
            "school":     np.random.choice(["GP", "MS"], n_samples),
            "sex":        np.random.choice(["F", "M"], n_samples),
            "age":        np.random.randint(15, 22, n_samples),
            "address":    np.random.choice(["U", "R"], n_samples),
            "famsize":    np.random.choice(["LE3", "GT3"], n_samples),
            "Pstatus":    np.random.choice(["T", "A"], n_samples),
            "Medu":       np.random.randint(0, 5, n_samples),
            "Fedu":       np.random.randint(0, 5, n_samples),
            "Mjob":       np.random.choice(["teacher","health","services","at_home","other"], n_samples),
            "Fjob":       np.random.choice(["teacher","health","services","at_home","other"], n_samples),
            "reason":     np.random.choice(["home","reputation","course","other"], n_samples),
            "guardian":   np.random.choice(["mother","father","other"], n_samples),
            "traveltime": np.random.randint(1, 5, n_samples),
            "studytime":  np.random.randint(1, 5, n_samples),
            "failures":   np.random.randint(0, 4, n_samples),
            "schoolsup":  np.random.choice(["yes","no"], n_samples),
            "famsup":     np.random.choice(["yes","no"], n_samples),
            "paid":       np.random.choice(["yes","no"], n_samples),
            "activities": np.random.choice(["yes","no"], n_samples),
            "nursery":    np.random.choice(["yes","no"], n_samples),
            "higher":     np.random.choice(["yes","no"], n_samples),
            "internet":   np.random.choice(["yes","no"], n_samples),
            "romantic":   np.random.choice(["yes","no"], n_samples),
            "famrel":     np.random.randint(1, 6, n_samples),
            "freetime":   np.random.randint(1, 6, n_samples),
            "goout":      np.random.randint(1, 6, n_samples),
            "Dalc":       np.random.randint(1, 6, n_samples),
            "Walc":       np.random.randint(1, 6, n_samples),
            "health":     np.random.randint(1, 6, n_samples),
            "absences":   np.random.randint(0, 30, n_samples),
            "G1":         np.random.randint(0, 21, n_samples),
            "G2":         np.random.randint(0, 21, n_samples),
            "G3":         np.random.randint(0, 21, n_samples),
        }

        df = pd.DataFrame(data)
        df["course"] = "demo"
        print(f"✅ Données de démo créées : {df.shape}")
        return df

    # ------------------------------------------------------------------
    def explore_data(self, df):
        print("\n🔍 Exploration des données")
        print("=" * 50)

        numeric_cols     = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        print(f"\n📈 Variables numériques ({len(numeric_cols)})  : {list(numeric_cols)}")
        print(f"🏷️  Variables catégorielles ({len(categorical_cols)}): {list(categorical_cols)}")

        print("\n📋 Statistiques descriptives:")
        print(df[numeric_cols].describe().round(2))

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        print("\n🔍 Valeurs manquantes:")
        print(missing if not missing.empty else "✅ Aucune valeur manquante")