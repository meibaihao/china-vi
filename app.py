import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Page Config
st.set_page_config(page_title="Vision Health Prediction Tool", layout="wide")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # Update these paths to match your exported filenames
    model = joblib.load('model_assets/best_model_lightgbm.pkl') # example name
    scaler = joblib.load('model_assets/scaler.pkl')
    encoders = joblib.load('model_assets/label_encoders.pkl')
    with open('model_assets/feature_list.txt', 'r') as f:
        features = [line.strip().split('. ')[1] for line in f.readlines() if '. ' in line]
    return model, scaler, encoders, features

try:
    model, scaler, encoders, feature_list = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- App Header ---
st.title("👓 Vision Health (Glass) Prediction Model")
st.markdown("""
This application uses a calibrated machine learning model to predict the likelihood of needing glasses 
based on demographic and health indicators.
""")

# --- Sidebar: Model Info ---
with st.sidebar:
    st.header("Model Statistics")
    st.info("Best Model: LightGBM (Calibrated)")
    st.write("Target: Glass Requirement")
    # You can manually input the threshold found in your model_info.txt
    OPTIMAL_THRESHOLD = 0.45 

# --- Input Form ---
st.header("Enter Subject Information")
col1, col2, col3 = st.columns(3)

user_inputs = {}

# We group inputs into columns for better UI
with col1:
    st.subheader("Demographics")
    user_inputs['gender'] = st.selectbox("Gender", ["1", "2"]) # Assuming 1/2 from training
    user_inputs['age'] = st.number_input("Age", min_value=0, max_value=120, value=65)
    user_inputs['rural'] = st.selectbox("Rural/Urban", ["1", "2"])
    user_inputs['edu'] = st.selectbox("Education Level", ["1", "2", "3", "4"])
    user_inputs['marry'] = st.selectbox("Marital Status", ["1", "2", "3"])

with col2:
    st.subheader("Health Metrics")
    user_inputs['bmi'] = st.number_input("BMI", value=24.0)
    user_inputs['srh'] = st.slider("Self-Reported Health (1-5)", 1, 5, 3)
    user_inputs['systo'] = st.number_input("Systolic BP", value=120)
    user_inputs['diasto'] = st.number_input("Diastolic BP", value=80)
    user_inputs['total_cognition'] = st.number_input("Cognition Score", value=20)

with col3:
    st.subheader("Medical History")
    # Standardize binary inputs
    binary_opts = ["0", "1"]
    user_inputs['hibpe'] = st.selectbox("Hypertension", binary_opts)
    user_inputs['diabe'] = st.selectbox("Diabetes", binary_opts)
    user_inputs['hearte'] = st.selectbox("Heart Disease", binary_opts)
    user_inputs['smokev'] = st.selectbox("Ever Smoked", binary_opts)
    user_inputs['exercise'] = st.selectbox("Regular Exercise", binary_opts)

# Fill in any missing features from the feature_list with defaults to avoid errors
for feat in feature_list:
    if feat not in user_inputs:
        user_inputs[feat] = 0

# --- Prediction Logic ---
if st.button("Predict Probability"):
    # 1. Create DataFrame
    input_df = pd.DataFrame([user_inputs])[feature_list]

    # 2. Process Categorical (Encoding)
    for col, le in encoders.items():
        if col in input_df.columns:
            # Handle unknown categories gracefully
            val = str(input_df[col].values[0])
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            else:
                input_df[col] = 0 # Default for unknown

    # 3. Process Numerical (Scaling)
    # Note: If your model is a tree model (LGBM/XGB), it might not need scaling,
    # but since you exported a scaler, we apply it.
    input_scaled = scaler.transform(input_df)

    # 4. Predict
    prob = model.predict_proba(input_scaled)[:, 1][0]
    prediction = 1 if prob >= OPTIMAL_THRESHOLD else 0

    # --- Display Results ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Predicted Probability", value=f"{prob:.2%}")
        if prediction == 1:
            st.error("Result: High Likelihood of Glass Requirement")
        else:
            st.success("Result: Low Likelihood of Glass Requirement")
            
    with res_col2:
        # Visual Gauge
        st.progress(prob)
        st.write(f"Decision Threshold used: {OPTIMAL_THRESHOLD}")

st.markdown("---")
st.caption("Disclaimer: This tool is for research purposes only and not for clinical diagnosis.")
