import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")

# Conversion Rate (1 USD to INR)
USD_TO_INR = 83.0 

@st.cache_data
def load_and_train():
    try:
        # Load the CSV
        df = pd.read_csv("vechile dataset.csv")
        df.columns = df.columns.str.strip()
        
        # Data Cleaning
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['price', 'mileage', 'year', 'make'])
        
        # Encoding
        le_dict = {}
        df_ml = df.copy()
        cat_cols = ['make', 'model', 'fuel', 'transmission', 'body']
        
        for col in cat_cols:
            le = LabelEncoder()
            # Force to string to prevent float vs str sorting/training errors
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le
            
        # Model Training
        features = ['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body']
        X = df_ml[features]
        y = df_ml['price']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, le_dict, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Run the training logic
model, le_dict, df = load_and_train()

if model is not None:
    st.title("🚗 Vehicle Price Predictor")
    st.write("Enter details below to get the predicted price in USD and INR.")

    col1, col2 = st.columns(2)
    
    with col1:
        # Convert to str before unique() to prevent the sort error
        makes = sorted(df['make'].astype(str).unique())
        make = st.selectbox("Manufacturer", makes)
        
        models_filt = df[df['make'] == make]['model'].astype(str).unique()
        model_name = st.selectbox("Model", sorted(models_filt))
        
        year = st.number_input("Year", 1990, 2026, 2024)
        
    with col2:
        mileage = st.number_input("Mileage", min_value=0, value=15000)
        
        fuels = sorted(df['fuel'].astype(str).unique())
        fuel = st.selectbox("Fuel Type", fuels)
        
        trans = sorted(df['transmission'].astype(str).unique())
        transmission = st.selectbox("Transmission", trans)
        
        bodies = sorted(df['body'].astype(str).unique())
        body = st.selectbox("Body Type", bodies)

    st.markdown("---")
    if st.button("Predict Price", type="primary"):
        # Prepare input
        input_data = pd.DataFrame([[
            le_dict['make'].transform([make])[0],
            le_dict['model'].transform([model_name])[0],
            year,
            mileage,
            le_dict['fuel'].transform([fuel])[0],
            le_dict['transmission'].transform([transmission])[0],
            le_dict['body'].transform([body])[0]
        ]], columns=['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body'])
        
        # Calculation
        pred_usd = model.predict(input_data)[0]
        pred_inr = pred_usd * USD_TO_INR
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Price in USD", value=f"${pred_usd:,.2f}")
        with res_col2:
            st.metric(label="Price in INR", value=f"₹{pred_inr:,.2f}")
        
        st.info(f"Conversion Rate Used: 1 USD = {USD_TO_INR} INR")