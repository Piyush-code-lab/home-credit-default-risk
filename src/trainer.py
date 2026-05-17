import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def get_features_and_target(df: pd.DataFrame):
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, config: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["evaluation"]["test_size"],
        random_state=config["evaluation"]["random_state"],
        stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def drop_low_importance_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_drop: list
):
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    print(f"Features after dropping low importance: {X_train.shape[1]}")
    return X_train, X_test


def build_model(config: dict) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        max_depth=config["model"]["max_depth"],
        min_child_weight=config["model"]["min_child_weight"],
        subsample=config["model"]["subsample"],
        colsample_bytree=config["model"]["colsample_bytree"],
        scale_pos_weight=config["model"]["scale_pos_weight"],
        random_state=config["model"]["random_state"],
        eval_metric="auc",
        n_jobs=-1
    )
    return model


def train_model(model: XGBClassifier, X_train, y_train) -> XGBClassifier:
    model.fit(X_train, y_train)
    print("Model training complete")
    return model


def save_model(model: XGBClassifier, model_path: str):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def save_feature_columns(feature_columns: list, features_path: str):
    with open(features_path, "w") as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {features_path}")


def load_model(model_path: str) -> XGBClassifier:
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_feature_columns(features_path: str) -> list:
    with open(features_path, "r") as f:
        feature_columns = json.load(f)
    return feature_columns