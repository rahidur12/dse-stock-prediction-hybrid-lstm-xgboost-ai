# =============================
# interference/predict_gp.py
# =============================

import pandas as pd
import numpy as np
import os
import joblib
import importlib
from tensorflow.keras.models import load_model

def get_prediction(symbol="gp"):
    # --- 1. SETUP PATHS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data", f"{symbol.lower()}_final_dataset.csv")

    # --- 2. LOAD ARTIFACTS ---
    try:
        # Load the Hybrid LSTM and XGBoost models
        lstm = load_model(os.path.join(model_dir, f"{symbol.lower()}_lstm_model.keras"))
        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        
        # Load the separate scalers (matches your train.py output)
        scaler_x = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))
        
        df = pd.read_csv(data_path)
    except Exception as e:
        return f"Error loading models/data: {e}", None

    # --- 3. PREPARE INPUT FEATURES ---
    # Features MUST match the order and count used in training
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']
    
    # Check if we have enough data (training used 1 step lookback in your specific run_experiment)
    if len(df) < 1:
        return "Error: Not enough data for prediction", None

    # Get the most recent row for prediction
    recent_data = df[features].tail(1).values 
    
    # Scale features using the X scaler
    scaled_data = scaler_x.transform(recent_data)
    
    # Reshape for LSTM (samples, time_steps, features) -> (1, 1, 8)
    X_lstm = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

    # --- 4. HYBRID INFERENCE ---
    # Step 1: LSTM Trend Prediction
    lstm_val_scaled = lstm.predict(X_lstm, verbose=0)
    
    # Step 2: XGBoost Refinement
    # In your training logic, XGBoost was trained on the 2D features (X_train_2d)
    # So we pass the scaled 2D features to XGBoost to get the refined prediction
    final_scaled_pred = xgb_model.predict(scaled_data)

    # --- 5. INVERSE SCALING ---
    # Convert back from (0, 1) range to actual BDT price using the Y scaler
    # XGBoost output is already 1D, so we reshape it for the scaler
    final_price = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1))[0][0]
    
    # Get the last known closing price for the 'delta' calculation in UI
    last_close = df['Close'].iloc[-1]
    
    return round(float(final_price), 2), round(float(last_close), 2)

if __name__ == "__main__":
    # Quick debug test
    price, last = get_prediction("gp")
    print(f"Latest Prediction: {price} BDT | Last Close: {last} BDT"),import streamlit as st