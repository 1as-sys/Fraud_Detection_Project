import streamlit as st
import numpy as np
import pickle
import os

st.title("Fraud Detection App")

# Absolute path to model
MODEL_PATH = r"C:\Users\HP\Downloads astha\Fraud_Detection_Project\models\rf_model.pkl"

# Load trained model
with open(MODEL_PATH, "rb") as f:
    rf = pickle.load(f)

st.success("✅ Model loaded successfully")

# User inputs
time = st.number_input("Transaction Time")
amount = st.number_input("Transaction Amount")

if st.button("Predict"):
    # 28 PCA features filled with 0 (demo purpose)
    input_data = np.array([time, amount] + [0]*28).reshape(1, -1)

    prediction = rf.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe")
