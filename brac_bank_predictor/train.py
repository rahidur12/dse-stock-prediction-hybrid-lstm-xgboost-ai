# =============================
# train.py
# =============================

import numpy as np
import tensorflow as tf
import random
import os
import pickle
import joblib

from data_preprocessing import load_and_prepare_data, prepare_datasets
from model_evaluation import run_experiment

# Seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def train_bracbank():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, "models")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data = load_and_prepare_data()
    if data is None:
        return

    merged_data, features_with_news, features_without_news = data

    data_n = prepare_datasets(merged_data, features_with_news)
    data_p = prepare_datasets(merged_data, features_without_news)

    res_news, actual_n, pred_n, lstm_model, xgb_model = run_experiment(*data_n[:-1], "News")

    scaler_y = data_n[-2]
    scaler_x = data_n[-1]

    lstm_model.save(os.path.join(model_dir, "bracbank_lstm_model.h5"))
    joblib.dump(xgb_model, os.path.join(model_dir, "bracbank_xgb_model.pkl"))

    with open(os.path.join(model_dir, "bracbank_scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(model_dir, "bracbank_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    res_plain, _, _, _, _ = run_experiment(*data_p[:-1], "Plain")

    final_results = {**res_news, **res_plain}

    print("\n📊 Final Results Summary")
    print(f"{'Model':<15} | {'RMSE':<10} | {'MAPE (%)':<10} | {'Correlation':<12}")
    print("-" * 55)

    for model_name, metrics in final_results.items():
        print(f"{model_name:<15} | {metrics[0]:.4f}   | {metrics[1]:.2f}       | {metrics[2]:.4f}")

    print(f"\n✅ Artifacts saved to: {model_dir}")


if __name__ == "__main__":
    train_bracbank()