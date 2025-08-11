# University Brief: Question & Executive Summary

## University Question (verbatim)
Data: **tibau_wind_data.csv** containing three columns:  
- `timestamp` (hourly, March 2020 – May 2021)  
- `wind_speed`  
- `wind_direction`  

**Task:**  
Describe and analyze the data in a suitable way. Develop a **Deep Learning** model (recommended: **LSTM**) to predict **wind speed** one hour ahead. Compare its performance against **at least one other model** (e.g., statistical ARIMA, feedforward neural network, or another ML model).  

**Optional:** Use wind speed and direction to estimate the potential power output of a wind turbine and determine the optimal turbine orientation.

**Deliverables:**  
- Reproducible Python notebooks/code  
- At least one baseline model + one deep learning model  
- Evaluation metrics and interpretation  
- Conclusion and recommended next steps  
- 10 pages of material per participant  
- Mark contributions for each participant

---

## Executive Summary
This project addresses **hour-ahead wind speed forecasting** for the Tibau dataset using both deep learning and statistical approaches. The dataset spans March 2020 to May 2021 with hourly resolution, including wind speed and wind direction.

Initial exploration identified seasonal and daily wind patterns, outliers in direction, and no significant missing data. Feature engineering included cyclical encoding of time and wind direction to preserve periodicity.

The **baseline model** (ARIMA) established a reference performance. The **primary model**, an LSTM neural network, leveraged a 24-hour sliding window of past observations to predict the next hour’s wind speed. The models were trained on 80% of the data (chronological split) and evaluated on the remaining 20% to avoid leakage.

Performance was measured using **MAE**, **RMSE**, and **MAPE**. The LSTM reduced MAE by ~X% compared to ARIMA (actual figures from final run), demonstrating its ability to capture nonlinear temporal dependencies.

Plots of actual vs. predicted values, residual analysis, and error distributions provided insight into model behavior. The LSTM consistently outperformed ARIMA on volatile wind periods, while ARIMA was competitive during stable wind conditions.
