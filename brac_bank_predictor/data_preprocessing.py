# =============================
# data_preprocessing.py
# =============================

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "bracbank_final_dataset.csv")

    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Run the agent pipeline first.")
        return None

    merged_data = pd.read_csv(data_path)
    merged_data.columns = merged_data.columns.str.strip()
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    merged_data = merged_data.sort_values("Date").reset_index(drop=True)

    features_with_news = ['Open', 'High', 'Low', 'Close', 'Volume', 'vader_score', 'vader_lag1', 'vader_lag2']
    features_without_news = ['Open', 'High', 'Low', 'Close', 'Volume']

    return merged_data, features_with_news, features_without_news


def prepare_datasets(df, feature_list, y_col='Close'):
    X = df[feature_list].values
    y = df[[y_col]].values

    train_idx = int(len(df) * 0.8)
    val_idx = int(len(df) * 0.9)

    X_train_raw, y_train_raw = X[:train_idx], y[:train_idx]
    X_val_raw, y_val_raw = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test_raw, y_test_raw = X[val_idx:], y[val_idx:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(X_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    X_test = scaler_x.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)

    X_train_3d = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val_3d = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_test_3d = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return (X_train_3d, X_val_3d, X_test_3d,
            X_train, X_val, X_test,
            y_train, y_val, y_test, scaler_y, scaler_x)