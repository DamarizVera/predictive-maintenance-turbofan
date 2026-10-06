"""
Modelos y métricas para predecir la vida útil remanente (RUL).

- Métricas: RMSE, MAE y el NASA Scoring Function (PHM08), que castiga más
  las predicciones tardías (sobrestimar la RUL es más peligroso que subestimarla).
- build_lstm: red LSTM apilada en TensorFlow/Keras.

TensorFlow se importa dentro de las funciones para que el resto del proyecto
(EDA, baselines) funcione aunque TensorFlow no esté instalado.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred) -> float:
    """
    Scoring function del desafío PHM08 (Saxena et al., 2008).

    d = predicción - real.
      d < 0 (predicción temprana):  exp(-d/13) - 1
      d >= 0 (predicción tardía):   exp(d/10) - 1
    Menor es mejor. Es asimétrica: predecir la falla tarde se castiga más.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred, name: str = "modelo") -> dict:
    """Resumen de métricas en un diccionario (útil para armar tablas comparativas)."""
    return {
        "modelo": name,
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "NASA_score": round(nasa_score(y_true, y_pred), 1),
    }


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    """Fija semillas para que el entrenamiento sea reproducible."""
    from tensorflow import keras

    # Fija Python, NumPy y TensorFlow en una sola llamada
    keras.utils.set_random_seed(seed)


def build_lstm(window: int, n_features: int, lstm_units=(64, 32),
               dropout: float = 0.2, learning_rate: float = 1e-3,
               output_scale: float = 125.0):
    """
    LSTM apilada para regresión de RUL.

    Entrada: (window, n_features) -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout
             -> Dense(16, relu) -> Dense(1) -> Rescaling(output_scale)

    La capa final Rescaling multiplica la salida por `output_scale` (el tope de RUL).
    Así la red trabaja internamente en el rango [0, 1] y entrega la RUL en ciclos.
    Sin ella, la red debe producir valores de hasta 125 desde activaciones tanh
    acotadas en [-1, 1]: las LSTM se saturan y el modelo colapsa a predecir
    un valor constante (la media de la RUL).
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="lstm_rul")
    model.add(keras.Input(shape=(window, n_features)))
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1))
    model.add(layers.Rescaling(output_scale))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def default_callbacks(patience: int = 10, checkpoint_path: str | None = None):
    """EarlyStopping (restaura los mejores pesos) + ReduceLROnPlateau (+ checkpoint opcional)."""
    from tensorflow import keras

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 2), min_lr=1e-5),
    ]
    if checkpoint_path:
        cbs.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss",
                                                   save_best_only=True))
    return cbs
