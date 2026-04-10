# =============================
# model_evaluation.py
# =============================

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=350, return_sequences=False),
        Dropout(0.3),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_experiment(X_train_3d, X_val_3d, X_test_3d,
                   X_train_2d, X_val_2d, X_test_2d,
                   y_train, y_val, y_test, scaler_y, label):

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    lstm_m = create_lstm_model((X_train_3d.shape[1], X_train_3d.shape[2]))
    lstm_m.fit(X_train_3d, y_train,
               validation_data=(X_val_3d, y_val),
               epochs=50, batch_size=32,
               verbose=0, callbacks=[early_stop])

    xg_m = train_xgboost(X_train_2d, y_train, X_val_2d, y_val)

    p_lstm = scaler_y.inverse_transform(lstm_m.predict(X_test_3d, verbose=0))
    p_xgb = scaler_y.inverse_transform(xg_m.predict(X_test_2d).reshape(-1, 1))
    actual = scaler_y.inverse_transform(y_test)

    def get_metrics(act, pred):
        rmse = np.sqrt(mean_squared_error(act, pred))
        mape = mean_absolute_percentage_error(act, pred) * 100
        corr = np.corrcoef(act.flatten(), pred.flatten())[0, 1]
        return rmse, mape, corr

    results = {
        f'LSTM-{label}': get_metrics(actual, p_lstm),
        f'XGB-{label}': get_metrics(actual, p_xgb)
    }

    return results, actual, p_lstm, lstm_m, xg_m