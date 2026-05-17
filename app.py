import json
import numpy as np
import pandas as pd
import gradio as gr
import shap
import joblib
import matplotlib.pyplot as plt
from src.trainer import load_model, load_feature_columns


# -----------------------------------------
# LOAD MODEL AND FEATURES
# -----------------------------------------
model = load_model("models/credit_risk_model_final.pkl")
feature_columns = load_feature_columns("models/feature_columns_final.json")
explainer = shap.TreeExplainer(model)


# -----------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------
def predict_default(
    amt_income,
    amt_credit,
    amt_annuity,
    age,
    employed_years,
    ext_source_1,
    ext_source_2,
    ext_source_3,
    late_payment_ratio,
    debt_to_credit_ratio,
    refused_apps,
    credit_utilization
):
    # Build input dict with zeros for all features
    input_data = {col: 0 for col in feature_columns}

    # Fill in provided values
    input_data["AMT_INCOME_TOTAL"] = amt_income
    input_data["AMT_CREDIT"] = amt_credit
    input_data["AMT_ANNUITY"] = amt_annuity
    input_data["age"] = age
    input_data["EMPLOYED_YEARS"] = employed_years
    input_data["EXT_SOURCE_1"] = ext_source_1
    input_data["EXT_SOURCE_2"] = ext_source_2
    input_data["EXT_SOURCE_3"] = ext_source_3
    input_data["LATE_PAYMENT_RATIO"] = late_payment_ratio
    input_data["DEBT_TO_CREDIT_RATIO"] = debt_to_credit_ratio
    input_data["REFUSED_APPS"] = refused_apps
    input_data["MEAN_UTILIZATION_RATIO"] = credit_utilization

    # Derived features
    input_data["CREDIT_INCOME_RATIO"] = amt_credit / amt_income if amt_income > 0 else 0
    input_data["ANNUITY_INCOME_RATIO"] = amt_annuity / amt_income if amt_income > 0 else 0
    input_data["CREDIT_TERM"] = amt_credit / amt_annuity if amt_annuity > 0 else 0
    input_data["DAYS_EMPLOYED_RATIO"] = employed_years / age if age > 0 else 0
    input_data["AGE_INCOME_RATIO"] = age / amt_income if amt_income > 0 else 0

    # Create dataframe
    input_df = pd.DataFrame([input_data])

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    risk_color = "red" if prob >= 0.5 else "green"

    # SHAP explanation
    shap_values = explainer.shap_values(input_df)
    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

    # Plot SHAP
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["red" if v > 0 else "green" for v in shap_df["SHAP Value"]]
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 10 factors influencing this prediction")
    ax.set_xlabel("SHAP Value (positive = higher default risk)")
    plt.tight_layout()

    result = f"Default Probability: {prob:.2%}\nRisk Level: {risk_label}"
    return result, fig


# -----------------------------------------
# GRADIO INTERFACE
# -----------------------------------------
with gr.Blocks(title="Credit Risk Predictor") as demo:
    gr.Markdown("# Credit Risk Predictor")
    gr.Markdown("Enter applicant details to predict default probability with explanation.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Financial Information")
            amt_income = gr.Number(label="Annual Income", value=150000)
            amt_credit = gr.Number(label="Credit Amount", value=500000)
            amt_annuity = gr.Number(label="Annuity Amount", value=25000)

            gr.Markdown("### Credit History")
            ext_source_1 = gr.Slider(0, 1, value=0.5, label="External Score 1")
            ext_source_2 = gr.Slider(0, 1, value=0.5, label="External Score 2")
            ext_source_3 = gr.Slider(0, 1, value=0.5, label="External Score 3")

        with gr.Column():
            gr.Markdown("### Personal Information")
            age = gr.Slider(20, 70, value=35, label="Age (years)")
            employed_years = gr.Slider(0, 40, value=5, label="Years Employed")

            gr.Markdown("### Risk Indicators")
            late_payment_ratio = gr.Slider(0, 1, value=0.0, label="Late Payment Ratio")
            debt_to_credit_ratio = gr.Slider(0, 1, value=0.3, label="Debt to Credit Ratio")
            refused_apps = gr.Slider(0, 10, value=0, step=1, label="Previous Refused Applications")
            credit_utilization = gr.Slider(0, 1, value=0.3, label="Credit Utilization Ratio")

    predict_btn = gr.Button("Predict Default Risk", variant="primary")

    with gr.Row():
        result_text = gr.Textbox(label="Prediction Result")
        shap_plot = gr.Plot(label="SHAP Explanation")

    predict_btn.click(
        fn=predict_default,
        inputs=[
            amt_income, amt_credit, amt_annuity,
            age, employed_years,
            ext_source_1, ext_source_2, ext_source_3,
            late_payment_ratio, debt_to_credit_ratio,
            refused_apps, credit_utilization
        ],
        outputs=[result_text, shap_plot]
    )

if __name__ == "__main__":
    demo.launch()