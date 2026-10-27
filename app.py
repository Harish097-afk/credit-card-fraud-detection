import os

import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    load_artifacts,
    load_threshold,
    prepare_inference_frame,
)
from review_policy import prioritize_reviews

MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", "ensemble_model.joblib")
SCALER_PATH = os.getenv("FRAUD_SCALER_PATH", "scaler.joblib")
THRESHOLD_PATH = os.getenv("FRAUD_THRESHOLD_PATH", "threshold.joblib")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "Soft-voting ensemble with probability-threshold classification "
    "and capacity-based risk prioritization."
)


@st.cache_resource
def load_model_and_scaler():
    return load_artifacts(MODEL_PATH, SCALER_PATH)


@st.cache_data
def get_default_threshold():
    return load_threshold(THRESHOLD_PATH)


model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error(
        "Model artifacts are missing. Train the model with: "
        "`python train_model.py --data-path path/to/creditcard.csv`"
    )
else:
    default_threshold = get_default_threshold()
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio(
        "Select Input Mode", ["Single Transaction", "Batch CSV Upload"]
    )

    if mode == "Single Transaction":
        fraud_threshold = st.sidebar.slider(
            "Fraud Probability Threshold",
            min_value=0.01,
            max_value=0.99,
            value=min(0.99, max(0.01, default_threshold)),
            step=0.01,
            help="A transaction is flagged when its predicted fraud probability "
            "meets or exceeds this threshold.",
        )
        st.subheader("Single Transaction Predictor")
        amount = st.number_input(
            "Transaction Amount", min_value=0.0, value=150.0, step=0.01
        )
        st.caption("Enter the V1–V28 features for the transaction.")
        feature_inputs = {
            feature: st.number_input(
                feature, value=0.0, step=0.001, format="%.6f"
            )
            for feature in FEATURE_COLUMNS
        }
        if st.button("Analyze Transaction", type="primary"):
            try:
                row = {AMOUNT_COLUMN: amount, **feature_inputs}
                prepared = prepare_inference_frame(pd.DataFrame([row]), scaler)
                probability = float(model.predict_proba(prepared)[0, 1])
                prediction = int(probability >= fraud_threshold)
                st.metric("Fraud risk score", f"{probability * 100:.2f}/100")
                if prediction:
                    st.error(f"FRAUD FLAGGED — probability: {probability:.2%}")
                else:
                    st.success(f"No fraud flag — probability: {probability:.2%}")
                st.caption(f"Decision threshold: {fraud_threshold:.2f}")
            except (ValueError, KeyError) as exc:
                st.error(f"Could not analyze transaction: {exc}")

    else:
        st.subheader("Batch CSV Inference")
        st.write(
            "Upload a CSV with V1–V28 and either raw `Amount` "
            "or pre-scaled `scaled_amount`."
        )
        policy = st.radio(
            "Batch decision policy",
            ["Probability threshold", "Top-risk review capacity"],
            horizontal=True,
            help="Threshold mode flags every transaction above a probability cutoff. "
            "Capacity mode prioritizes a fixed percentage of the highest-risk rows.",
        )
        if policy == "Probability threshold":
            fraud_threshold = st.slider(
                "Fraud Probability Threshold",
                min_value=0.01,
                max_value=0.99,
                value=min(0.99, max(0.01, default_threshold)),
                step=0.01,
            )
        else:
            review_capacity = st.select_slider(
                "Transactions to prioritize for review",
                options=[0.5, 1.0, 2.0, 5.0, 10.0],
                value=0.5,
                format_func=lambda x: f"{x:g}%",
            )
            st.caption(
                "The app prioritizes the highest-risk rows up to this capacity. "
                "This ranks transactions; it does not change the model."
            )

        uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview", data.head(3))
                missing_features = [
                    c for c in FEATURE_COLUMNS if c not in data.columns
                ]
                if missing_features:
                    st.error(f"Missing required feature columns: {missing_features}")
                elif AMOUNT_COLUMN not in data.columns and "scaled_amount" not in data.columns:
                    st.error("The CSV must include either 'Amount' or 'scaled_amount'.")
                elif st.button("Run Predictions", type="primary"):
                    eval_frame = prepare_inference_frame(data, scaler)
                    probabilities = model.predict_proba(eval_frame)[:, 1]

                    result = data.copy()
                    result["Fraud_Probability"] = probabilities
                    result["Risk_Score"] = (probabilities * 100).round(2)

                    if policy == "Probability threshold":
                        predictions = (probabilities >= fraud_threshold).astype(int)
                        result["Fraud_Prediction"] = predictions
                        result["Fraud_Label"] = [
                            "Fraud" if prediction else "Legitimate"
                            for prediction in predictions
                        ]
                        result["Decision_Threshold"] = fraud_threshold
                        result["Review_Priority"] = np.where(
                            predictions == 1, "Flagged by threshold", "Normal"
                        )
                        st.metric("Flagged Transactions", int(predictions.sum()))
                        st.metric("Flagged Rate", f"{predictions.mean():.2%}")
                    else:
                        priority_mask = prioritize_reviews(
                            probabilities, review_capacity
                        )
                        result["Review_Priority"] = np.where(
                            priority_mask, "Priority Review", "Normal"
                        )
                        st.metric("Transactions prioritized", int(priority_mask.sum()))
                        st.metric(
                            "Review capacity",
                            f"{priority_mask.mean():.2%}",
                        )
                        st.info(
                            "Priority Review means selected for investigation, "
                            "not confirmed fraud."
                        )

                    result = result.sort_values(
                        "Fraud_Probability", ascending=False
                    ).reset_index(drop=True)
                    st.success(f"Analysis complete for {len(result):,} transactions.")
                    st.dataframe(result.head(25), use_container_width=True)
                    csv_bytes = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download all predictions (CSV)",
                        data=csv_bytes,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                        type="primary",
                    )
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                st.error(f"Batch prediction failed: {exc}")
