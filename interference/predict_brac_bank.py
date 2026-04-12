# =============================
# interference/predict_gp.py (or predict_brac_bank.py)
# =============================

import pandas as pd
import numpy as np
import os       # <--- THIS WAS MISSING
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def get_prediction(symbol="gp"):
    # --- 1. SETUP PATHS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data", f"{symbol.lower()}_final_dataset.csv")

    # --- 2. LOAD ARTIFACTS ---
    try:
        # Define the exact architecture used in your training
        # 1 time step, 8 features (Open, High, Low, Close, Volume, vader, lag1, lag2)
        model = Sequential([
            Input(shape=(1, 8)), 
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        # Load the weights manually
        model_path = os.path.join(model_dir, f"{symbol.lower()}_lstm_model.h5")
        model.load_weights(model_path)
        
        # Load XGBoost and Scalers
        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        scaler_x = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))
        
        df = pd.read_csv(data_path)
        
    except Exception as e:
        # This will now show clearly in your app's debug console
        return f"Prediction Script Error: {str(e)}", 0.0

    # --- 3. PREPARE INPUT FEATURES ---
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']
    
    if len(df) < 1:
        return "Error: Not enough data", 0.0

    recent_data = df[features].tail(1).values 
    scaled_data = scaler_x.transform(recent_data)
    X_lstm = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

    # --- 4. HYBRID INFERENCE ---
    # Use the 'model' we manually built and loaded weights into
    # We ignore the LSTM trend and go straight to the XGBoost final prediction 
    # as per your previous logic
    final_scaled_pred = xgb_model.predict(scaled_data)

    # --- 5. INVERSE SCALING ---
    final_price = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1))[0][0]
    last_close = df['Close'].iloc[-1]
    
    return round(float(final_price), 2), round(float(last_close), 2)

if __name__ == "__main__":
    # Quick debug test
    price, last = get_prediction("bracbank")
    print(f"Latest Prediction: {price} BDT | Last Close: {last} BDT")