# =============================
# interference/predict_brac_bank.py
# =============================
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf


def get_prediction(symbol="bracbank", live_entry=None):
    # --- 1. SETUP PATHS ---
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir    = os.path.join(project_root, "models")
    data_path    = os.path.join(project_root, "data", f"{symbol.lower()}_final_dataset.csv")

    # --- 2. LOAD ARTIFACTS ---
    try:
        model_path = os.path.join(model_dir, f"{symbol.lower()}_lstm_model.h5")

        # Load full saved Keras model from .h5 (architecture + weights together)
        model = tf.keras.models.load_model(model_path, compile=False)

        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        scaler_x  = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y  = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))

        df = pd.read_csv(data_path)

    except Exception as e:
        return f"Model Loading Error: {str(e)}", 0.0

    # --- 3. PREPARE INPUT FEATURES ---
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']

    if live_entry:
        # Take price/volume from DSE Live, sentiment from the most recent CSV records
        live_data_row = {
            'Open':        live_entry['Open'],
            'High':        live_entry['High'],
            'Low':         live_entry['Low'],
            'Close':       live_entry['Close'],
            'Volume':      live_entry['Volume'],
            'vader_score': df['vader_score'].iloc[-1],
            'vader_lag1':  df['vader_score'].iloc[-2],
            'vader_lag2':  df['vader_score'].iloc[-3]
        }
        recent_data = np.array([[live_data_row[f] for f in features]])
        last_close  = live_entry['Close']
    else:
        # Fallback to last CSV row
        if len(df) < 1:
            return "Error: No data in CSV", 0.0
        recent_data = df[features].tail(1).values
        last_close  = df['Close'].iloc[-1]

    # --- 4. HYBRID INFERENCE ---
    scaled_data = scaler_x.transform(recent_data)

    # XGBoost Prediction
    final_scaled_pred = xgb_model.predict(scaled_data)

    # --- 5. INVERSE SCALING ---
    final_price = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1))[0][0]

    return round(float(final_price), 2), round(float(last_close), 2)