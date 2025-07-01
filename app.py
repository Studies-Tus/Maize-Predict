import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("maize_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature columns
feature_columns = [
    'Rainfall (mm)', 'Temperature (°C)', 'Soil_pH', 'Soil_Clay (%)',
    'Soil_Silt (%)', 'Soil_Sand (%)', 'Fertilizer_Use (kg/ha)',
    'Planting_Density (plants/m²)'
]

st.title("🌽 Maize Yield Prediction App")
st.write("Enter the values below to predict maize yield (t/ha).")

# Input fields for user
user_input = []
for feature in feature_columns:
    value = st.number_input(f"{feature}", value=0.0, format="%.4f")
    user_input.append(value)

if st.button("Predict Yield"):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.success(f"🌾 Predicted Maize Yield: {prediction:.2f} t/ha")
