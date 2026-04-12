import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def get_prediction(symbol="gp"):
    # --- 1. SETUP PATHS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data", f"{symbol.lower()}_final_dataset.csv")

    # --- 2. LOAD ARTIFACTS ---
    try:
        # EXACT MATCH to your model_evaluation.py
        model = Sequential([
            Input(shape=(1, 8)), 
            LSTM(units=350, return_sequences=False), # Changed 64 -> 350
            Dropout(0.3),                            # Added Dropout layer
            Dense(units=1)
        ])
        
        model_path = os.path.join(model_dir, f"{symbol.lower()}_lstm_model.h5")
        
        # This will now find all 3 layers it expects
        model.load_weights(model_path)
        
        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        scaler_x = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))
        
        df = pd.read_csv(data_path)
        
    except Exception as e:
        return f"Architecture Error: {str(e)}", 0.0

    # --- 3. PREPARE INPUT FEATURES ---
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']
    
    if len(df) < 1:
        return "Error: Not enough data", 0.0

    recent_data = df[features].tail(1).values 
    scaled_data = scaler_x.transform(recent_data)

    # --- 4. HYBRID INFERENCE ---
    # XGBoost Prediction (using the 2D scaled data)
    final_scaled_pred = xgb_model.predict(scaled_data)

    # --- 5. INVERSE SCALING ---
    final_price = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1))[0][0]
    last_close = df['Close'].iloc[-1]
    
    return round(float(final_price), 2), round(float(last_close), 2)