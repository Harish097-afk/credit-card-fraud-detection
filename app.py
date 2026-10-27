import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("Replication of *Khalid et al. (2024)* — **SMOTE + Soft-Voting Ensemble Framework**")

@st.cache_resource
def load_models():
    if os.path.exists("ensemble_model.joblib") and os.path.exists("scaler.joblib"):
        model = joblib.load("ensemble_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    return None, None

model, scaler = load_models()

if model is None:
    st.error("⚠️ Model artifacts missing. Upload 'ensemble_model.joblib' and 'scaler.joblib' to the repository.")
else:
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Select Input Mode", ["Single Transaction", "Batch CSV Upload"])

    if mode == "Single Transaction":
        st.subheader("⚙️ Single Transaction Predictor")
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
        with col2:
            st.info("Features V1 to V28 default to 0.0 for testing.")

        v_features = {f"V{i}": 0.0 for i in range(1, 29)}

        if st.button("Analyze Transaction"):
            scaled_amt = scaler.transform([[amount]])[0][0]
            input_dict = {**v_features, 'scaled_amount': scaled_amt}
            input_df = pd.DataFrame([input_dict])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** Probability: {prob * 100:.2f}%")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION.** Fraud Probability: {prob * 100:.2f}%")

    elif mode == "Batch CSV Upload":
        st.subheader("📁 Batch File Inference")
        uploaded_file = st.file_uploader("Upload CSV containing V1-V28 and Amount", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Sample Preview:", data.head(3))

            if st.button("Run Predictions"):
                df_proc = data.copy()
                if 'Amount' in df_proc.columns:
                    df_proc['scaled_amount'] = scaler.transform(df_proc['Amount'].values.reshape(-1, 1))
                    df_proc.drop(['Amount', 'Time'], axis=1, errors='ignore', inplace=True)
                
                if 'Class' in df_proc.columns:
                    X_eval = df_proc.drop('Class', axis=1)
                else:
                    X_eval = df_proc

                preds = model.predict(X_eval)
                probs = model.predict_proba(X_eval)[:, 1]

                data['Fraud_Prediction'] = preds
                data['Fraud_Probability'] = probs

                st.success("Analysis Complete!")
                st.write(data[['Fraud_Prediction', 'Fraud_Probability']].head(10))
                st.metric("Total Flagged Frauds", int(np.sum(preds == 1)))
