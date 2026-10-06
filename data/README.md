# Data / Datos

**EN** — This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset (Saxena & Goebel, 2008). The raw files are not tracked in git.

**ES** — Este proyecto usa el dataset **NASA C-MAPSS Turbofan Engine Degradation Simulation** (Saxena & Goebel, 2008). Los archivos no se versionan en git.

## Download / Descarga

1. Download the zip from the PHM Society mirror of the NASA Prognostics Data Repository / Descarga el zip desde el espejo de la PHM Society:
   https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
   (Alternative / alternativa: Kaggle, "NASA Turbofan Jet Engine Data Set")
2. Unzip it (it may contain a second zip, `CMAPSSData.zip`) / Descomprímelo (puede traer un segundo zip adentro, `CMAPSSData.zip`).
3. Copy the `.txt` files into `data/raw/` / Copia los `.txt` a `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── ... (FD002–FD004)
```

## Citation / Cita

A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
