# TIBAU Wind Forecasting — Assignment-Ready Repo

Clean, reproducible pipeline for **hour-ahead wind speed forecasting** on `tibau_wind_data.csv` using:
- **Baseline:** ARIMA (or Persistence) for comparison
- **Deep Learning:** LSTM sequence model
- **CRISP-DM**-aligned structure, metrics, and plots

## Quickstart
```bash
# 0) Create env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# 1) EDA (optional but recommended)
python src/eda.py --csv data/tibau_wind_data.csv

# 2) Train baseline (ARIMA)
python src/train_baseline.py --csv data/tibau_wind_data.csv --target wind_speed

# 3) Train LSTM
python src/train_lstm.py --csv data/tibau_wind_data.csv --target wind_speed --window 24 --epochs 20

# 4) Evaluate & compare (prints table)
python src/evaluate.py --csv data/tibau_wind_data.csv --target wind_speed --window 24
```

Artifacts (saved under `artifacts/`):
- `baseline.pkl` — ARIMA model
- `lstm.keras` — Keras LSTM model
- `metrics.json` — test metrics for both models
- Plots under `artifacts/plots/`

## Data
- Input: `data/tibau_wind_data.csv` with columns:
  `timestamp` (hourly), `wind_speed`, `wind_direction`

> If your CSV uses a different delimiter/locale, pass `--sep` and `--decimal` to the scripts.

## Structure
```
tibau-wind-forecasting-assignment/
├─ data/
│  └─ tibau_wind_data.csv              # (included if you uploaded it here)
├─ src/
│  ├─ eda.py                           # describe data, plots, correlations
│  ├─ utils.py                         # common helpers
│  ├─ train_baseline.py                # ARIMA baseline
│  ├─ train_lstm.py                    # LSTM training
│  └─ evaluate.py                      # side-by-side metrics
├─ artifacts/                          # saved models + plots
├─ QUESTION_SUMMARY.md                 # paste/keep uni brief + your summary
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ README.md
└─ .github/workflows/python-ci.yml
```

## Notes
- LSTM uses a sliding window of past `window` hours to predict the next hour wind speed.
- ARIMA order is auto-selected with a quick heuristic; adjust if needed.
- Keep your **university prompt** inside `QUESTION_SUMMARY.md` so reviewers see the mapping from task → code.
