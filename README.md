🇬🇧 English | [🇪🇸 Español](README.es.md)

# Predictive Maintenance: Remaining Useful Life of Turbofan Engines with LSTM

> 🚧 **Work in progress** — exploratory analysis complete; baselines and LSTM model under development.

## Project Overview

This project predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines from multivariate sensor time series, using a **Long Short-Term Memory (LSTM)** neural network built with TensorFlow/Keras. The goal is to support **condition-based maintenance**: scheduling interventions based on the actual health of each engine instead of fixed calendars or after failures occur.

The LSTM is benchmarked against classical machine learning models (Random Forest, XGBoost) trained on the same information, so the choice of a deep learning model is justified by results, not assumed.

---

## Objective

- Model engine degradation from 21 sensor channels recorded cycle by cycle until failure.
- Predict how many operating cycles each engine has left before failure.
- Compare a sequence model (LSTM) with tabular baselines using RMSE, MAE and the asymmetric NASA scoring function, which penalizes late predictions more heavily than early ones.

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation** (Saxena & Goebel, 2008), the standard benchmark for prognostics and predictive maintenance. Each engine starts healthy with unknown initial wear and develops a fault until it fails.

| Subset | Train / test engines | Operating conditions | Fault modes |
|---|---|---|---|
| **FD001** (this project) | 100 / 100 | 1 | 1 (HPC degradation) |

Data is not tracked in git — see [`data/README.md`](data/README.md) for download instructions.

---

## Methodology

1. **Exploratory analysis** — data quality, engine lifetimes, uninformative sensors, degradation trends aligned by cycles-to-failure, correlation with RUL.
2. **Target definition** — piecewise-linear RUL capped at 125 cycles (healthy early life is not observable in the sensors).
3. **Leakage-safe preparation** — train/validation split **by engine**, Min-Max scaling fitted on training data only, 30-cycle sliding windows.
4. **Baselines** — Random Forest and XGBoost on window features (last value, rolling mean, rolling std, slope).
5. **LSTM** — stacked LSTM (64 → 32 units) with dropout, early stopping and learning-rate scheduling.
6. **Evaluation** — RMSE, MAE and NASA score on the official test set.

---

## Results

*Coming soon.*

---

## Project Structure

```
predictive-maintenance-turbofan/
├── data/
│   ├── raw/                       # C-MAPSS .txt files (not tracked)
│   └── README.md                  # download instructions
├── notebooks/
│   └── 01_exploracion_datos.ipynb # exploratory data analysis
├── src/
│   ├── data_processing.py         # loading, RUL, scaling, sequences, window features
│   └── models.py                  # metrics (RMSE, MAE, NASA score) and LSTM
├── models/                        # trained models (not tracked)
├── reports/figures/               # exported charts
├── requirements.txt
└── README.md / README.es.md
```

---

## How to Run

```bash
git clone https://github.com/DamarizVera/predictive-maintenance-turbofan.git
cd predictive-maintenance-turbofan
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# download the data into data/raw/ (see data/README.md)
jupyter notebook notebooks/
```

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · Matplotlib · Seaborn

---

## Reference

A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *International Conference on Prognostics and Health Management (PHM08)*, Denver, CO, 2008.

---

## Author

**Damariz Vera** — Data Science & Business Intelligence
[github.com/DamarizVera](https://github.com/DamarizVera)
