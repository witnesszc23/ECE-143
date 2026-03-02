from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Configuration
data_path = Path("../student_performance.csv").expanduser() # CHANGE this directory to your dataset!!!
model_save_dir = Path("./saved_models").expanduser() # CHANGE this to your desired output directory!!!
random_state = 42
test_size = 0.2
n_iter_search = 100

actionable_cols = [
    "daily_study_hours", "attendance_percentage", "motivation_score",
    "sleep_hours", "screen_time_hours", "physical_activity_minutes",
    "homework_completion_rate", "exam_anxiety_score", "study_environment"
]

def load_and_split_data(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Loads dataset, filters features, and splits into train/test sets."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {filepath}. Please verify the path.")
        raise

    valid_cols = [c for c in actionable_cols if c in df.columns]
    X_action = df[valid_cols].copy()
    
    y_reg = df["final_score"]
    y_clf = df["pass_fail"]

    print(f"Loaded data successfully. Using features: {valid_cols}")

    return train_test_split(
        X_action, y_reg, y_clf, 
        test_size=test_size, 
        random_state=random_state
    )

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Builds a scikit-learn preprocessing ColumnTransformer dynamically."""
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")

    return ColumnTransformer(
        transformers=[
            # sparse_output=False is required because SHAP prefers dense arrays
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

def get_model_config() -> Dict[str, Dict[str, Any]]:
    """Returns the configuration for all models, their parameters, and task types."""
    return {
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "task": "classification",
            "scoring": "accuracy",
            "params": {
                "model__n_neighbors": randint(3, 30),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            }
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=random_state),
            "task": "regression",
            "scoring": "neg_mean_absolute_error",
            "params": {
                "model__n_estimators": randint(50, 200),
                "model__max_depth": randint(5, 50),
                "model__min_samples_split": randint(2, 20),
            }
        },
        "XGBoost": {
            "estimator": XGBRegressor(random_state=random_state, objective='reg:squarederror'),
            "task": "regression",
            "scoring": "neg_mean_absolute_error",
            "params": {
                "model__n_estimators": randint(50, 300),
                "model__learning_rate": uniform(0.01, 0.2),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.5, 0.5),
                "model__colsample_bytree": uniform(0.5, 0.5)
            }
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "task": "regression",
            "scoring": "neg_mean_absolute_error",
            "params": {
                "model__n_estimators": randint(50, 300),
                "model__learning_rate": uniform(0.01, 0.2),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.5, 0.5)
            }
        },
    }

