# University Brief: Question & Executive Summary

## University Question (verbatim)
- Data file: **tibau_wind_data.csv** containing hourly `timestamp` (Mar 2020–May 2021), `wind speed`, and `wind direction`.
- **Task:** Describe/analyze the data and **build a Deep Learning model** (recommended: **LSTM**) to **predict wind speed one hour ahead**. 
  Compare performance against **at least one other model** (e.g., **ARIMA**, feedforward NN, or ML model).
- **Optional:** Use wind speed & direction to estimate turbine power output and discuss optimal orientation.
- **Deliverables:** Reproducible Python notebooks/code; analysis; evaluation; conclusion and next steps.

## Executive Summary (fill in 8–12 lines in your own words)
- Goal: Hour-ahead **wind speed** forecasting on Tibau data (hourly).
- Baselines: ARIMA (statistical) and Persistence; Main model: **LSTM**.
- Data checks: coverage, gaps/outliers, distributions, correlations; basic cleaning.
- Features: pure univariate (wind speed) and multivariate (add wind direction as sin/cos).
- Split: time-based (train = first 80%, test = last 20%). No leakage.
- Metrics: MAE, RMSE, MAPE on test period.
- Result (example placeholders — replace with your runs): LSTM ↓MAE vs ARIMA by ~X%; stable generalization.
- Plots: EDA visuals, prediction vs. actual, residuals.
- Outlook: hyperparameter tuning, exogenous weather, power curve mapping.

*(Replace the placeholder bullets with your actual numbers and comments.)*
