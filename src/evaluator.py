import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model(model, X_test, y_test) -> float:
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"ROC-AUC Score: {auc:.4f}")
    return auc


def plot_roc_curve(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()
    print("ROC curve saved.")


def plot_confusion_matrix(model, X_test, y_test, threshold: float = 0.5):
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


def get_shap_importance(model, X_test) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    return importance, shap_values


def plot_shap_summary(shap_values, X_test, max_display: int = 20):
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        max_display=max_display
    )
    shap.summary_plot(
        shap_values,
        X_test,
        max_display=max_display
    )


def get_low_importance_features(
    importance_df: pd.DataFrame,
    threshold: float = 0.001
) -> list:
    cols_to_drop = importance_df[
        importance_df["importance"] < threshold
    ]["feature"].tolist()
    print(f"Low importance features to drop: {len(cols_to_drop)}")
    return cols_to_drop