[🇬🇧 English](README.md) | 🇪🇸 Español

# Mantenimiento predictivo: vida útil remanente de motores turbofan con LSTM

> 🚧 **En desarrollo** — análisis exploratorio terminado; baselines y modelo LSTM en construcción.

## Descripción del proyecto

Este proyecto predice la **vida útil remanente (RUL, *Remaining Useful Life*)** de motores turbofan de avión a partir de series de tiempo multivariadas de sensores, usando una red neuronal **LSTM (*Long Short-Term Memory*)** construida con TensorFlow/Keras. El objetivo es apoyar el **mantenimiento basado en condición**: programar intervenciones según el estado real de cada motor, en vez de calendarios fijos o después de la falla.

La LSTM se compara contra modelos clásicos de machine learning (Random Forest, XGBoost) entrenados con la misma información, de modo que la elección de un modelo de deep learning se justifica con resultados y no se da por supuesta.

---

## Objetivo

- Modelar la degradación de motores a partir de 21 sensores registrados ciclo a ciclo hasta la falla.
- Predecir cuántos ciclos de operación le quedan a cada motor antes de fallar.
- Comparar un modelo secuencial (LSTM) con baselines tabulares usando RMSE, MAE y la función de puntaje asimétrica de la NASA, que castiga más las predicciones tardías que las tempranas.

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation** (Saxena & Goebel, 2008), el benchmark estándar de pronóstico de fallas y mantenimiento predictivo. Cada motor parte sano, con un desgaste inicial desconocido, y desarrolla una falla hasta dejar de operar.

| Subconjunto | Motores train / test | Condiciones de operación | Modos de falla |
|---|---|---|---|
| **FD001** (este proyecto) | 100 / 100 | 1 | 1 (degradación del HPC) |

Los datos no se versionan en git — ver [`data/README.md`](data/README.md) para las instrucciones de descarga.

---

## Metodología

1. **Análisis exploratorio** — calidad de datos, vida útil de los motores, sensores sin información, tendencias de degradación alineadas por ciclos antes de la falla, correlación con la RUL.
2. **Definición del objetivo** — RUL lineal por tramos con tope de 125 ciclos (la vida temprana sana no es observable en los sensores).
3. **Preparación sin fuga de datos** — separación train/validación **por motor**, escalado Min-Max ajustado solo con train, ventanas deslizantes de 30 ciclos.
4. **Baselines** — Random Forest y XGBoost con features de ventana (último valor, media móvil, desviación móvil, pendiente).
5. **LSTM** — LSTM apilada (64 → 32 unidades) con dropout, early stopping y ajuste de tasa de aprendizaje.
6. **Evaluación** — RMSE, MAE y NASA score sobre el conjunto de prueba oficial.

---

## Resultados

*Próximamente.*

---

## Estructura del proyecto

```
predictive-maintenance-turbofan/
├── data/
│   ├── raw/                       # archivos .txt de C-MAPSS (no versionados)
│   └── README.md                  # instrucciones de descarga
├── notebooks/
│   └── 01_exploracion_datos.ipynb # análisis exploratorio
├── src/
│   ├── data_processing.py         # carga, RUL, escalado, secuencias, features de ventana
│   └── models.py                  # métricas (RMSE, MAE, NASA score) y LSTM
├── models/                        # modelos entrenados (no versionados)
├── reports/figures/               # gráficos exportados
├── requirements.txt
└── README.md / README.es.md
```

---

## Cómo ejecutarlo

```bash
git clone https://github.com/DamarizVera/predictive-maintenance-turbofan.git
cd predictive-maintenance-turbofan
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# descargar los datos en data/raw/ (ver data/README.md)
jupyter notebook notebooks/
```

---

## Tecnologías

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · Matplotlib · Seaborn

---

## Referencia

A. Saxena, K. Goebel, D. Simon y N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *International Conference on Prognostics and Health Management (PHM08)*, Denver, CO, 2008.

---

## Autora

**Damariz Vera** — Data Science & Business Intelligence
[github.com/DamarizVera](https://github.com/DamarizVera)
