"""
Carga y preparación de los datos NASA C-MAPSS (Turbofan Engine Degradation Simulation).

Funciones principales:
- load_cmapss: lee los archivos train / test / RUL de un subconjunto (FD001..FD004).
- add_train_rul / add_test_rul: calcula la vida útil remanente (RUL) por ciclo.
- find_low_variance_columns: detecta sensores sin información (constantes).
- split_units: separa motores en entrenamiento / validación (sin fuga de datos).
- make_sequences / make_test_sequences: ventanas deslizantes para la LSTM.
- make_window_features: features tabulares (media, desviación, pendiente) para baselines.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Esquema del dataset
# ---------------------------------------------------------------------------

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Descripción de los 21 sensores (Saxena et al., 2008, Tabla 2)
SENSOR_DESCRIPTIONS = {
    "s_1": "T2 - Temperatura entrada fan (°R)",
    "s_2": "T24 - Temperatura salida LPC (°R)",
    "s_3": "T30 - Temperatura salida HPC (°R)",
    "s_4": "T50 - Temperatura salida LPT (°R)",
    "s_5": "P2 - Presión entrada fan (psia)",
    "s_6": "P15 - Presión ducto bypass (psia)",
    "s_7": "P30 - Presión total salida HPC (psia)",
    "s_8": "Nf - Velocidad física del fan (rpm)",
    "s_9": "Nc - Velocidad física del núcleo (rpm)",
    "s_10": "epr - Razón de presión del motor",
    "s_11": "Ps30 - Presión estática salida HPC (psia)",
    "s_12": "phi - Razón flujo combustible / Ps30",
    "s_13": "NRf - Velocidad corregida del fan (rpm)",
    "s_14": "NRc - Velocidad corregida del núcleo (rpm)",
    "s_15": "BPR - Razón de bypass",
    "s_16": "farB - Razón combustible/aire del quemador",
    "s_17": "htBleed - Entalpía de sangrado",
    "s_18": "Nf_dmd - Velocidad fan demandada (rpm)",
    "s_19": "PCNfR_dmd - Velocidad fan corregida demandada (rpm)",
    "s_20": "W31 - Sangrado refrigerante HPT (lbm/s)",
    "s_21": "W32 - Sangrado refrigerante LPT (lbm/s)",
}

# Tope de RUL usado en la literatura para FD001 (RUL "por tramos"):
# al inicio de su vida el motor está sano y la RUL real no es observable
# en los sensores, por lo que se limita a un valor máximo constante.
DEFAULT_RUL_CAP = 125


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> pd.DataFrame:
    """Lee un archivo de C-MAPSS (separado por espacios, sin encabezado)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Algunas versiones traen columnas vacías al final por espacios extra
    df = df.dropna(axis=1, how="all")
    return df


def load_cmapss(data_dir, subset: str = "FD001"):
    """
    Carga train, test y RUL real del subconjunto indicado.

    Parameters
    ----------
    data_dir : str o Path
        Carpeta que contiene train_FD00X.txt, test_FD00X.txt y RUL_FD00X.txt.
    subset : str
        "FD001", "FD002", "FD003" o "FD004".

    Returns
    -------
    train : DataFrame  (series completas hasta la falla)
    test : DataFrame   (series truncadas antes de la falla)
    rul_test : DataFrame con columnas [unit, RUL_end] (RUL real al último ciclo de test)
    """
    data_dir = Path(data_dir)
    train = _read_txt(data_dir / f"train_{subset}.txt")
    test = _read_txt(data_dir / f"test_{subset}.txt")
    rul = _read_txt(data_dir / f"RUL_{subset}.txt")

    train.columns = COLUMNS
    test.columns = COLUMNS
    rul.columns = ["RUL_end"]
    rul.insert(0, "unit", np.arange(1, len(rul) + 1))

    for df in (train, test):
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    return train, test, rul


# ---------------------------------------------------------------------------
# Variable objetivo: RUL
# ---------------------------------------------------------------------------

def add_train_rul(train: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """
    Agrega la RUL a cada fila de entrenamiento: ciclos que faltan para la falla.

    En train cada motor corre hasta fallar, así que RUL = último ciclo - ciclo actual.
    Si `cap` no es None, se agrega también la columna RUL_capped (RUL limitada).
    """
    df = train.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if cap is not None:
        df["RUL_capped"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(test: pd.DataFrame, rul_test: pd.DataFrame,
                 cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """
    Agrega la RUL real a cada fila de test.

    RUL_test.txt entrega la RUL al ÚLTIMO ciclo observado de cada motor; para un ciclo
    anterior la RUL es RUL_end + (último ciclo observado - ciclo actual).
    Se usa solo para análisis; el modelo se evalúa en el último ciclo de cada motor.
    """
    df = test.merge(rul_test, on="unit", how="left")
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["RUL_end"] + (last_cycle - df["cycle"])
    df = df.drop(columns="RUL_end")
    if cap is not None:
        df["RUL_capped"] = df["RUL"].clip(upper=cap)
    return df


# ---------------------------------------------------------------------------
# Selección de variables
# ---------------------------------------------------------------------------

def find_low_variance_columns(df: pd.DataFrame, cols=None, threshold: float = 1e-2) -> list:
    """Columnas cuya desviación estándar es menor a `threshold` (no aportan información)."""
    cols = cols if cols is not None else SETTING_COLS + SENSOR_COLS
    std = df[cols].std()
    return std[std < threshold].index.tolist()


def get_feature_columns(train: pd.DataFrame, threshold: float = 1e-2) -> list:
    """Sensores con variación suficiente para modelar (calculado solo con train)."""
    drop = set(find_low_variance_columns(train, SENSOR_COLS, threshold))
    return [c for c in SENSOR_COLS if c not in drop]


# ---------------------------------------------------------------------------
# Separación y escalado
# ---------------------------------------------------------------------------

def split_units(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Separa por MOTOR (no por fila) en entrenamiento y validación.

    Separar por fila filtraría información: ciclos consecutivos del mismo motor
    quedarían en ambos conjuntos y la validación sería optimista.
    """
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=max(1, int(len(units) * val_frac)), replace=False)
    is_val = df["unit"].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy()


def fit_scaler(train: pd.DataFrame, feature_cols: list) -> MinMaxScaler:
    """Ajusta un MinMaxScaler SOLO con datos de entrenamiento."""
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


# ---------------------------------------------------------------------------
# Secuencias para la LSTM
# ---------------------------------------------------------------------------

def make_sequences(df: pd.DataFrame, feature_cols: list, window: int = 30,
                   target: str = "RUL_capped"):
    """
    Ventanas deslizantes por motor.

    Cada muestra X[i] son `window` ciclos consecutivos de sensores (window, n_features)
    y y[i] es la RUL en el último ciclo de esa ventana.
    Motores con menos de `window` ciclos se omiten.
    """
    X_list, y_list = [], []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target].to_numpy(dtype=np.float32)
        n = len(g)
        if n < window:
            continue
        for end in range(window, n + 1):
            X_list.append(values[end - window:end])
            y_list.append(targets[end - 1])
    X = np.stack(X_list) if X_list else np.empty((0, window, len(feature_cols)), np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def make_test_sequences(df: pd.DataFrame, feature_cols: list, window: int = 30):
    """
    Última ventana de cada motor de test (es donde se evalúa la RUL).

    Si un motor tiene menos de `window` ciclos, se rellena al inicio repitiendo
    su primer registro (padding), para no descartarlo.
    """
    X_list, units = [], []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = g[feature_cols].to_numpy(dtype=np.float32)
        if len(values) < window:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            values = np.vstack([pad, values])
        X_list.append(values[-window:])
        units.append(unit)
    return np.stack(X_list), np.array(units)


# ---------------------------------------------------------------------------
# Features tabulares para baselines (XGBoost / Random Forest)
# ---------------------------------------------------------------------------

def make_window_features(df: pd.DataFrame, feature_cols: list, window: int = 30) -> pd.DataFrame:
    """
    Resume la historia reciente de cada sensor con features tabulares:
    valor actual, media y desviación móvil, y pendiente (tendencia) en la ventana.

    Permite comparar la LSTM contra modelos clásicos usando la MISMA información
    (los últimos `window` ciclos).
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")
    out = df[INDEX_COLS].copy()

    # Pendiente de una regresión lineal sobre la ventana: cov(t, x) / var(t)
    def _slope(x):
        if len(x) < 2:
            return 0.0
        tc = np.arange(len(x), dtype=float)
        tc -= tc.mean()
        return float((tc * (x - x.mean())).sum() / (tc ** 2).sum())

    for col in feature_cols:
        roll = g[col].rolling(window, min_periods=1)
        out[f"{col}_last"] = df[col]
        out[f"{col}_mean"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{col}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        out[f"{col}_slope"] = (
            roll.apply(_slope, raw=True).reset_index(level=0, drop=True)
        )

    for extra in ("RUL", "RUL_capped"):
        if extra in df.columns:
            out[extra] = df[extra]
    return out
