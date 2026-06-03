import sys
import os
from pathlib import Path
import tensorflow as tf

# =====================================================
# 🔥 PATH CONFIG
# =====================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(ROOT)

from brain_tumor_classifier.models.efficient import build_efficientnet

print("✅ PATH configurado:", ROOT)

# =====================================================
# 📂 RUTAS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TEST_PATH = DATA_DIR / "Testing"

print(f"📁 Test dir: {TEST_PATH}")

# =====================================================
# ⚙️ PARÁMETROS
# =====================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# =====================================================
# 📥 DATASET TEST
# =====================================================

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = test_ds.class_names

# =====================================================
# 🔄 NORMALIZACIÓN
# =====================================================

norm = tf.keras.layers.Rescaling(1./255)

test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 🤖 CARGAR MODELO
# =====================================================

model = tf.keras.models.load_model("models/efficientnet.keras")

# =====================================================
# 📊 EVALUACIÓN
# =====================================================

loss, acc = model.evaluate(test_ds)

print(f"\n📊 TEST LOSS: {loss:.4f}")
print(f"✅ TEST ACCURACY: {acc:.4f}")