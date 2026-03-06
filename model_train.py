import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_model():
    # 1. Load data
    df = pd.read_csv("vechile dataset.csv")
    
    # Clean column names (removes hidden spaces)
    df.columns = df.columns.str.strip()

    # 2. Data Cleaning
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Drop rows where essential data is missing
    df = df.dropna(subset=['price', 'mileage', 'year', 'make', 'model'])

    # 3. Features and Target
    features = ['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'body']
    
    # 4. Encoding
    le_dict = {}
    df_ml = df.copy()
    categorical_cols = ['make', 'model', 'fuel', 'transmission', 'body']

    for col in categorical_cols:
        le = LabelEncoder()
        # We convert to string and then fit
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        le_dict[col] = le

    # 5. Train Model
    X = df_ml[features]
    y = df_ml['price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 6. Save Files
    joblib.dump(model, 'vehicle_model.pkl')
    joblib.dump(le_dict, 'encoders.pkl')
    print("Training Complete. Created 'vehicle_model.pkl' and 'encoders.pkl'")

if __name__ == "__main__":
    train_and_save_model()