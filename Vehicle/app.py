import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# ---------------------------------------------
# Load the Trained Model
# ---------------------------------------------
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_vehicle_price_model.cbm")
    return model

model = load_model()

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("🚗 Vehicle Price Prediction App")
st.write("Enter vehicle details below to estimate the price.")

# Dropdown options 
makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Mercedes', 'Hyundai', 'Unknown']
fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'E85 Flex Fuel', 'PHEV Hybrid Fuel']
transmissions = ['Automatic', 'Variable', 'CVT', 'A/T', 'Unknown']
bodies = ['SUV', 'Pickup Truck', 'Sedan', 'Passenger Van', 'Cargo Van', 'Hatchback', 'Convertible', 'Minivan']
drivetrain_types = ['Four-wheel Drive', 'All-wheel Drive', 'Rear-wheel Drive', 'Front-wheel Drive']

# ---------------------------------------------
# Input Fields
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacture Year", min_value=1980, max_value=2025, value=2018)
    mileage = st.number_input("Mileage (miles)", min_value=0, max_value=6000, value=50)
    cylinders = st.number_input("Engine Cylinders", min_value=2, max_value=16, value=4)
    doors = st.selectbox("Number of Doors", [2, 3, 4, 5], index=2)

with col2:
    make = st.text_input("Make (e.g., Toyota, Ford)", "Toyota")
    model_name = st.text_input("Model (e.g., Camry, Fiesta)", "Camry")
    engine = st.text_input("Engine Description (e.g., 16V PDI DOHC Turbo, DOHC 16V, Multiair variable valve)", "16V PDI DOHC Turbo")
    fuel = st.selectbox("Fuel Type", fuel_types)
    transmission = st.selectbox("Transmission", transmissions)
    body = st.selectbox("Body Style", bodies)
    drivetrain = st.selectbox("Drivetrain", drivetrain_types)

# ---------------------------------------------
# Prepare Input for Prediction
# ---------------------------------------------
def prepare_input():
    data = {
        "year": year,
        "mileage": mileage,
        "cylinders": cylinders,
        "doors": doors,
        "make": make,
        "model": model_name,
        "engine": engine,
        "fuel": fuel,
        "transmission": transmission,
        "body": body,
        "drivetrain": drivetrain
    }
    return pd.DataFrame([data])

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
if st.button("Predict Price"):
    input_df = prepare_input()

    # Predict price
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Vehicle Price: **${prediction:,.2f}**")
