import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("ensemble_heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")  

st.set_page_config(page_title="Heart Disease Detection App", layout="centered")

st.title("❤️ Heart Disease Detection App")
st.write("Enter the following details to predict the likelihood of heart disease:")

# Collect user inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1=True, 0=False)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0, step=0.1)
slope = st.selectbox("Slope (1–3)", [1, 2, 3])

# Convert inputs into dataframe
input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data_scaled)
    prob = model.predict_proba(input_data_scaled)[0][prediction[0]]

    if prediction[0] == 1:
        st.error(f"💔 The model predicts **Heart Disease** with {prob*100:.2f}% confidence.")
    else:
        st.success(f"💚 The model predicts **No Heart Disease** with {prob*100:.2f}% confidence.")
