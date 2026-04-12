import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def get_prediction(symbol="gp"):
    # ... [Setup paths same as before] ...

    try:
        # 1. Manually define the architecture (Must match your training)
        model = Sequential([
            Input(shape=(1, 8)), # 1 time step, 8 features
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        # 2. Load ONLY the weights from your .h5 file
        # This ignores the 'InputLayer' config that is causing the crash
        model_path = os.path.join(model_dir, f"{symbol.lower()}_lstm_model.h5")
        model.load_weights(model_path)
        
        # 3. Load the other artifacts
        xgb_model = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_xgb_model.pkl"))
        scaler_x = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_x.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, f"{symbol.lower()}_scaler_y.pkl"))
        
        df = pd.read_csv(data_path)
        
        # Replace 'lstm' with 'model' in your prediction logic below
        lstm = model 
        
    except Exception as e:
        return f"Weights Load Error: {str(e)}", None

    # ... [Rest of your prediction logic] ...

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
    price, last = get_prediction("bracbank")
    print(f"Latest Prediction: {price} BDT | Last Close: {last} BDT")