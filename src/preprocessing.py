from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Préprocesseur sklearn-compatible.
    Utilisé comme première étape du pipeline ImbPipeline dans modeling.py.
    """

    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.ordinal_cols = None

    # ------------------------------------------------------------------
    def _define_column_types(self, X: pd.DataFrame):
        numeric_cols = []
        categorical_cols = []
        ordinal_cols = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Entiers avec peu de valeurs distinctes → ordinal
                if X[col].nunique() <= 10 and X[col].dtype.kind in "iu":
                    ordinal_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        print("\n🔧 Classification des colonnes:")
        print(f"  Numériques   : {len(numeric_cols)}")
        print(f"  Catégorielles: {len(categorical_cols)}")
        print(f"  Ordinales    : {len(ordinal_cols)}")

        return numeric_cols, categorical_cols, ordinal_cols

    # ------------------------------------------------------------------
    def _build_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,   # sklearn >= 1.2 (remplace sparse=False)
            )),
        ])

        ordinal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
                ("ord", ordinal_transformer, self.ordinal_cols),
            ],
            remainder="drop",
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("❌ DataPreprocessor attend un DataFrame pandas")

        self.numeric_cols, self.categorical_cols, self.ordinal_cols = \
            self._define_column_types(X)

        self._build_preprocessor()
        self.preprocessor.fit(X, y)
        self._extract_feature_names()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("❌ DataPreprocessor attend un DataFrame pandas")
        return self.preprocessor.transform(X)

    # ------------------------------------------------------------------
    def _extract_feature_names(self):
        names = []
        names.extend(self.numeric_cols)

        if self.categorical_cols:
            ohe = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(self.categorical_cols))

        names.extend(self.ordinal_cols)
        self.feature_names = list(names)
        print(f"✅ {len(self.feature_names)} features après prétraitement")