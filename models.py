from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import MODELS_DIR
from .features import label_clusters


@dataclass
class TrainedModels:
    classifiers: Dict[str, Pipeline]
    metrics: pd.DataFrame
    cluster_frame: pd.DataFrame
    kmeans_model: KMeans


def _context_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["batting_team", "bowling_team", "venue"]
    numeric = [
        "target",
        "current_score",
        "runs_needed",
        "overs_remaining",
        "balls_remaining",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
    ]
    missing = [col for col in categorical + numeric if col not in frame.columns]
    if missing:
        raise ValueError(f"Training frame missing columns: {missing}")
    return categorical, numeric


def _baseline_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["venue", "team1", "team2", "toss_winner", "toss_decision"]
    numeric = ["first_innings_score", "toss_winner_is_team1"]
    missing = [col for col in categorical + numeric if col not in frame.columns]
    if missing:
        raise ValueError(f"Baseline frame missing columns: {missing}")
    return categorical, numeric


def _classification_pipeline(categorical: list[str], numeric: list[str], estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _fit_and_score(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    numeric: list[str],
    purpose: str,
) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    pipeline = _classification_pipeline(categorical, numeric, estimator)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return pipeline, {
        "model": name,
        "purpose": purpose,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def train_classifiers(context_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> tuple[Dict[str, Pipeline], pd.DataFrame]:
    baseline_categorical, baseline_numeric = _baseline_feature_columns(baseline_frame)
    baseline_X = baseline_frame[baseline_categorical + baseline_numeric]
    baseline_y = baseline_frame["team1_won"]

    context_categorical, context_numeric = _context_feature_columns(context_frame)
    context_X = context_frame[context_categorical + context_numeric]
    context_y = context_frame["won"]

    model_specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            baseline_X,
            baseline_y,
            baseline_categorical,
            baseline_numeric,
            "match_level_baseline",
        ),
        "random_forest": (
            RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
            ),
            context_X,
            context_y,
            context_categorical,
            context_numeric,
            "live_match_context",
        ),
        "xgboost": (
            XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            ),
            context_X,
            context_y,
            context_categorical,
            context_numeric,
            "live_match_context",
        ),
    }

    fitted = {}
    metrics = []
    for name, (estimator, X, y, categorical, numeric, purpose) in model_specs.items():
        pipeline, metric = _fit_and_score(name, estimator, X, y, categorical, numeric, purpose)
        fitted[name] = pipeline
        metrics.append(metric)
    return fitted, pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)


def train_player_clusters(cluster_input: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, dict]:
    features = [
        "seasons",
        "batting_runs",
        "batting_average",
        "strike_rate",
        "runs_per_match",
        "bowling_wickets",
        "economy",
        "wickets_per_match",
    ]
    X = cluster_input[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    labeled = cluster_input.copy()
    labeled["cluster_id"] = clusters
    labeled["cluster_name"] = label_clusters(cluster_input, clusters)
    wrapped = {"scaler": scaler, "model": kmeans}
    return labeled, wrapped


def save_artifacts(classifiers: Dict[str, Pipeline], metrics: pd.DataFrame, cluster_frame: pd.DataFrame, kmeans_model: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in classifiers.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_player_clusters.joblib")
    metrics.to_csv(MODELS_DIR / "model_metrics.csv", index=False)
    cluster_frame.to_csv(MODELS_DIR / "player_clusters.csv", index=False)


def load_classifier(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_model_metrics() -> pd.DataFrame:
    path = MODELS_DIR / "model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def predict_win_probability(model: Pipeline, payload: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(payload)[:, 1]
