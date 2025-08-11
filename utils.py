"""
Shared utilities: loading, feature engineering, metrics, windowing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_wind_csv(path: str, sep=",", decimal=".", parse_dates=True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, decimal=decimal)
    if parse_dates:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    return out


def add_winddir_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # wind_direction in degrees: encode as sin/cos
    rad = np.deg2rad(out["wind_direction"].astype(float))
    out["wind_dir_sin"] = np.sin(rad)
    out["wind_dir_cos"] = np.cos(rad)
    return out


def make_supervised(df: pd.DataFrame, target="wind_speed", window=24, horizon=1, feature_cols=None):
    """
    Turn a univariate/multivariate time series into supervised samples for LSTM.
    X shape: (samples, window, features) ; y shape: (samples,)
    """
    use = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in use.columns if c not in ["timestamp"]]
    X, y = [], []
    arr = use[feature_cols].values.astype(float)
    tgt = use[target].values.astype(float)
    for i in range(len(use) - window - horizon + 1):
        X.append(arr[i:i+window])
        y.append(tgt[i+window+horizon-1])
    return np.array(X), np.array(y), feature_cols


def time_split(n: int, train_ratio=0.8) -> Tuple[int, Tuple[slice, slice]]:
    split = int(n * train_ratio)
    return split, (slice(0, split), slice(split, n))


def metrics_regression(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true), eps))) * 100.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": mape}
