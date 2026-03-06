import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")

# Conversion rate (Example: 1 USD = 83 INR)
USD_TO_INR = 83.0

@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("vechile dataset.csv")
    except FileNotFoundError:
        st.error("Missing File: Please upload 'vechile dataset.csv' to your repository.")
        return None, None, None

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Basic Cleaning
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['price', 'mileage', 'year', 'make'])
    
    # Encoding Categorical Data
    le_dict = {}
    df_ml = df.copy()
    cat_cols = ['make', 'model', 'fuel', 'transmission', 'body']
    
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to handle potential NaN/float issues during training
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        le_dict[col] = le
        
    features = ['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body']
    X = df_ml[features]
    y = df_ml['price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_dict, df

# Execution
model, le_dict, df = load_and_train()

if model is not None:
    st.title("🚗 Vehicle Price Predictor")
    st.write("Enter the vehicle details below to get an estimated market price.")

    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Convert to string before sorting to prevent float vs str error
        make_options = sorted(df['make'].astype(str).unique())
        make = st.selectbox("Select Manufacturer", make_options)
        
        # FIX: Filtered model list also needs string conversion
        filtered_models = df[df['make'] == make]['model'].astype(str).unique()
        model_name = st.selectbox("Select Model", sorted(filtered_models))
        
        year = st.number_input("Manufacture Year", 1990, 2026, 2024)
        
    with col2:
        mileage = st.number_input("Mileage (miles)", min_value=0, value=10000)
        
        # FIX: Apply string conversion to remaining dropdowns
        fuel_options = sorted(df['fuel'].astype(str).unique())
        fuel = st.selectbox("Fuel Type", fuel_options)
        
        trans_options = sorted(df['transmission'].astype(str).unique())
        trans = st.selectbox("Transmission", trans_options)
        
        body_options = sorted(df['body'].astype(str).unique())
        body = st.selectbox("Body Style", body_options)

    if st.button("Predict Price", type="primary"):
        # Create input for prediction
        input_data = pd.DataFrame([[
            le_dict['make'].transform([make])[0],
            le_dict['model'].transform([model_name])[0],
            year,
            mileage,
            le_dict['fuel'].transform([fuel])[0],
            le_dict['transmission'].transform([trans])[0],
            le_dict['body'].transform([body])[0]
        ]], columns=['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body'])
        
        # Predict in USD
        prediction_usd = model.predict(input_data)[0]
        
        # Convert to INR
        prediction_inr = prediction_usd * USD_TO_INR
        
        st.success(f"### Estimated Price (USD): ${prediction_usd:,.2f}")
        st.success(f"### Estimated Price (INR): ₹{prediction_inr:,.2f}")