import shutil
from pathlib import Path

import kagglehub

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

    print("\nDescargando dataset desde KaggleHub...", flush=True)

    # Descargar dataset
    downloaded_path = kagglehub.dataset_download(
        KAGGLE_DATASET
    )

    downloaded_path = Path(downloaded_path)

    # Crear directorio destino
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copiar contenido descargado
    for item in downloaded_path.iterdir():

        destino = DATA_DIR / item.name

        if item.is_dir():

            shutil.copytree(
                item,
                destino,
                dirs_exist_ok=True
            )

        else:

            shutil.copy2(
                item,
                destino
            )

    print(
        "\nDataset descargado correctamente.",
        flush=True
    )


preparar_dataset_kaggle()