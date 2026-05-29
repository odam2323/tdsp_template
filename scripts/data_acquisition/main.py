import os
from pathlib import Path
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"


def preparar_dataset_kaggle():

    training_dir = DATA_DIR / "Training"
    testing_dir = DATA_DIR / "Testing"

    if training_dir.exists() and testing_dir.exists():
        print("\n✅ Dataset ya disponible.")
        return

    print("\n⬇️ Descargando dataset desde Kaggle...")

    # Descargar dataset
    os.system(f"kaggle datasets download -d {DATASET_NAME}")

    zip_path = ROOT_DIR / "brain-tumor-mri-dataset.zip"

    if not zip_path.exists():
        print("\n❌ Error descargando dataset.")
        return

    print("\n📦 Extrayendo dataset...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Opcional: borrar zip
    os.remove(zip_path)

    print("\n✅ Dataset descargado y listo.")


if __name__ == "__main__":
    preparar_dataset_kaggle()