import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal, GlorotUniform, Zeros

def get_prediction(symbol="gp"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data", f"{symbol.lower()}_final_dataset.csv")

    # --- THE CRITICAL FIX: Custom Object Mapping ---
    # This maps the "New" dictionary format to the "Actual" classes
    custom_objects = {
        "Orthogonal": Orthogonal,
        "GlorotUniform": GlorotUniform,
        "Zeros": Zeros
    }

    try:
        model_file = os.path.join(model_dir, f"{symbol.lower()}_lstm_model.keras")
        
        # Load with custom_objects to fix the 'Initializer Identifier' error
        lstm = load_model(
            model_file, 
            custom_objects=custom_objects, 
            compile=False
        )
        
        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        scaler_x = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))
        df = pd.read_csv(data_path)

        # --- PREDICTION LOGIC ---
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']
        recent_data = df[features].tail(1).values 
        scaled_data = scaler_x.transform(recent_data)
        X_lstm = np.reshape(scaled_data, (1, 1, 8))

        # Inference
        lstm_val = lstm.predict(X_lstm, verbose=0)
        final_scaled_pred = xgb_model.predict(scaled_data)
        
        final_price = scaler_y.inverse_transform(final_scaled_pred.reshape(-1, 1))[0][0]
        last_close = df['Close'].iloc[-1]
        
        return round(float(final_price), 2), round(float(last_close), 2)

    except Exception as e:
        return f"Model Error: {str(e)}", 0.0