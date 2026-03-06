import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")

# Load components
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('vehicle_model.pkl')
        le_dict = joblib.load('encoders.pkl')
        df = pd.read_csv("vechile dataset.csv")
        df.columns = df.columns.str.strip() # Clean column names
        return model, le_dict, df
    except Exception as e:
        st.error(f"Initialization Error: {e}. Did you run model_train.py?")
        return None, None, None

model, le_dict, df = load_assets()

if model is not None:
    st.title("🚗 Vehicle Price Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.selectbox("Make", sorted(df['make'].unique()))
        model_name = st.selectbox("Model", sorted(df[df['make'] == make]['model'].unique()))
        year = st.number_input("Year", 1990, 2026, 2024)
        
    with col2:
        mileage = st.number_input("Mileage", min_value=0, value=10000)
        fuel = st.selectbox("Fuel", df['fuel'].unique())
        trans = st.selectbox("Transmission", df['transmission'].unique())
        body = st.selectbox("Body", df['body'].unique())

    if st.button("Predict Price", type="primary"):
        # Create input row
        input_data = pd.DataFrame([[
            le_dict['make'].transform([make])[0],
            le_dict['model'].transform([model_name])[0],
            year,
            mileage,
            le_dict['fuel'].transform([fuel])[0],
            le_dict['transmission'].transform([trans])[0],
            le_dict['body'].transform([body])[0]
        ]], columns=['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body'])
        
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Price: ${prediction:,.2f}")