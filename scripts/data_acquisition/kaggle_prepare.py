import os
import sys
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / 'docs' / 'data' / 'datos'

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

def preparar_dataset_kaggle():

    training_dir = DATA_DIR / "Training"
    testing_dir = DATA_DIR / "Testing"

    # Dataset ya existe
    if training_dir.exists() and testing_dir.exists():

        print("\nDataset ya disponible.", flush=True)
        return

    print("\nPreparando dataset desde Kaggle...", flush=True)

    # =========================================
    # CONFIGURAR CREDENCIALES
    # =========================================

    kaggle_json = ROOT_DIR / "kaggle.json"

    if not kaggle_json.exists():

        print("\nERROR: kaggle.json no encontrado.", flush=True)

        sys.exit(1)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    shutil.copy(
        kaggle_json,
        kaggle_dir / "kaggle.json"
    )

    try:
        os.chmod(
            kaggle_dir / "kaggle.json",
            0o600
        )
    except:
        pass

    # =========================================
    # DESCARGAR DATASET
    # =========================================

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    api = KaggleApi()

    api.authenticate()

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DATA_DIR),
        unzip=True
    )

    print(
        "\nDataset descargado correctamente.",
        flush=True
    )