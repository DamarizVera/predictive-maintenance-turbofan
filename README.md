🇬🇧 English | [🇪🇸 Español](README.es.md)

# Predictive Maintenance: Remaining Useful Life of Turbofan Engines with LSTM

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
5. **LSTM** — stacked LSTM (64 → 32 units) with dropout, output rescaling, early stopping and learning-rate scheduling; final model as a 3-seed ensemble.
6. **Evaluation** — RMSE, MAE and NASA score on the official test set.

---

## Results

Evaluation on the official FD001 test set (100 engines, last observed cycle, RUL capped at 125):

| Model | RMSE | MAE | NASA score |
|---|---:|---:|---:|
| **XGBoost** (window features) | **12.57** | **9.66** | **237.2** |
| **LSTM** (3-model ensemble) | 12.81 | 9.93 | 256.5 |
| Random Forest (window features) | 13.53 | 10.47 | 296.1 |
| LSTM (single model, mean of 3 seeds) | 14.11 ± 0.68 | 10.59 | 326.9 |
| Ridge | 17.30 | 13.95 | 572.5 |
| Mean baseline | 41.94 | 34.83 | 33,354.5 |

![Model comparison](reports/figures/12_comparacion_modelos.png)

### Key findings

- **Equivalent performance:** the LSTM ensemble (RMSE 12.81) and XGBoost (12.57) differ by 0.24 cycles (1.9%). The LSTM reaches this level from raw sensor sequences, with no feature engineering; XGBoost relies on 56 hand-crafted window features.
- **Ensembling matters:** single LSTM models vary with the random seed (RMSE 14.11 ± 0.68). Averaging three models reduces the error by 9% and removes that dependency.
- **Accurate where it matters:** the LSTM's error is lowest for engines close to failure (MAE 4.3 cycles for RUL ≤ 25, 5.9 cycles for RUL 26–50), the range where maintenance decisions are made.
- **Model choice:** for FD001, XGBoost offers the lowest error, lower training cost and higher interpretability. The LSTM is a competitive alternative that does not depend on manual feature design, a property expected to matter more under multiple operating conditions (FD002/FD004).

![Sensor degradation trends](reports/figures/03_tendencias_sensores.png)

---

## Project Structure

```
predictive-maintenance-turbofan/
├── data/
│   ├── raw/                       # C-MAPSS .txt files (not tracked)
│   └── README.md                  # download instructions
├── notebooks/
│   ├── 01_exploracion_datos.ipynb # exploratory data analysis
│   ├── 02_baselines.ipynb         # Ridge, Random Forest, XGBoost
│   └── 03_lstm.ipynb              # LSTM and final comparison
├── src/
│   ├── data_processing.py         # loading, RUL, scaling, sequences, window features
│   └── models.py                  # metrics (RMSE, MAE, NASA score) and LSTM
├── models/                        # trained models (not tracked)
├── reports/
│   ├── figures/                   # exported charts
│   └── resultados_finales.csv     # final metrics
├── requirements.txt
└── README.md / README.es.md
```

---

## How to Run

```bash
git clone https://github.com/DamarizVera/predictive-maintenance-turbofan.git
cd predictive-maintenance-turbofan
python3.12 -m venv .venv   # TensorFlow supports Python 3.10–3.13
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
