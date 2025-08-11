"""
Baseline forecaster: ARIMA (hour-ahead). Saves model + metrics.
If ARIMA fails, falls back to persistence (y_hat_t = y_{t-1}).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import load_wind_csv, metrics_regression

def fit_arima(series):
    try:
        import statsmodels.api as sm
        # quick auto order selection (simple heuristic)
        # You can hardcode e.g. (2,1,2) if needed
        order = (2, 1, 2)
        model = sm.tsa.ARIMA(series, order=order).fit()
        return model
    except Exception as e:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/tibau_wind_data.csv")
    ap.add_argument("--target", type=str, default="wind_speed")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--decimal", type=str, default=".")
    args = ap.parse_args()

    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True, parents=True)

    df = load_wind_csv(args.csv, sep=args.sep, decimal=args.decimal)
    y = df[args.target].astype(float).values
    split = int(len(y)*args.train_ratio)
    y_train, y_test = y[:split], y[split:]

    model = fit_arima(y_train)
    if model is not None:
        pred = model.forecast(steps=len(y_test))
        y_hat = np.array(pred)
        model.save(str(artifacts/"baseline.pkl"))
        kind = "ARIMA(2,1,2)"
    else:
        # persistence fallback
        y_hat = np.roll(y_test, 1)
        y_hat[0] = y_train[-1]
        kind = "Persistence (t-1)"

    metrics = metrics_regression(y_test, y_hat)
    metrics_out = {"baseline": {"model": kind, **metrics}}
    (artifacts/"metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print("Baseline:", kind, "Metrics:", metrics)

if __name__ == "__main__":
    main()
