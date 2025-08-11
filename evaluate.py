"""
Loads artifacts/metrics.json (baseline + LSTM) and prints a small comparison table.
If not present, runs the minimal flow.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from utils import load_wind_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/tibau_wind_data.csv")
    ap.add_argument("--target", type=str, default="wind_speed")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--decimal", type=str, default=".")
    args = ap.parse_args()

    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        print("metrics.json not found. Run train_baseline.py and train_lstm.py first.")
        return

    d = json.loads(metrics_path.read_text())
    rows = []
    for k, v in d.items():
        rows.append({"Model": k, "Detail": v.get("model",""), "MAE": v.get("MAE"), "RMSE": v.get("RMSE"), "MAPE": v.get("MAPE")})
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())

if __name__ == "__main__":
    main()
