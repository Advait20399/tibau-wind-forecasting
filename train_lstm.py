"""
LSTM model for hour-ahead wind speed forecasting.

- Builds supervised sequences with window W (default 24 hours)
- Trains a small LSTM; saves model and metrics
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_wind_csv, add_time_features, add_winddir_cyclical, make_supervised, time_split, metrics_regression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/tibau_wind_data.csv")
    ap.add_argument("--target", type=str, default="wind_speed")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--decimal", type=str, default=".")
    args = ap.parse_args()

    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True, parents=True)
    plots = artifacts / "plots"; plots.mkdir(exist_ok=True, parents=True)

    # 1) Load + basic features
    df = load_wind_csv(args.csv, sep=args.sep, decimal=args.decimal)
    df = add_time_features(df)
    if "wind_direction" in df.columns:
        df = add_winddir_cyclical(df)

    # 2) Supervised windowing (multivariate by default)
    X, y, feature_cols = make_supervised(df, target=args.target, window=args.window, horizon=1, feature_cols=None)
    split, (tr, te) = time_split(len(X), train_ratio=0.8)
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    # 3) Build model
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    # 4) Evaluate
    y_hat = model.predict(X_test).ravel()
    metrics = metrics_regression(y_test, y_hat)

    # Save model + metrics
    model.save(artifacts/"lstm.keras")
    # Merge with baseline metrics if present
    metrics_path = artifacts/"metrics.json"
    if metrics_path.exists():
        prev = json.loads(metrics_path.read_text())
    else:
        prev = {}
    prev["lstm"] = {"model": "LSTM(64)->Dense", **metrics}
    metrics_path.write_text(json.dumps(prev, indent=2))
    print("LSTM Metrics:", metrics)

if __name__ == "__main__":
    main()
