import sys
import os
from pathlib import Path
import tensorflow as tf
import time

# =====================================================
# 🔥 PATH CONFIG
# =====================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(ROOT)

print("✅ PATH configurado:", ROOT)

from brain_tumor_classifier.models.efficient import build_efficientnet

# =====================================================
# ⏱️ INICIO
# =====================================================

start_total = time.time()
print("\n🚀 TRAINING PIPELINE EFFICIENTNET\n")

# =====================================================
# 📂 RUTAS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_PATH = DATA_DIR / "Training"

print(f"📁 Train dir: {TRAIN_PATH}")

# =====================================================
# ⚙️ PARÁMETROS
# =====================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 5

# =====================================================
# 📥 DATASET (TRAIN + VAL SPLIT)
# =====================================================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Clases: {class_names}")

# =====================================================
# 🔄 NORMALIZACIÓN
# =====================================================

norm = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 🤖 MODELO
# =====================================================

model = build_efficientnet(
    input_shape=(128, 128, 3),
    num_classes=num_classes
)

# =====================================================
# 💾 CALLBACKS
# =====================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =====================================================
# 🚀 TRAINING
# =====================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

model.save("models/efficientnet_savedmodel")

print("\n🏁 TRAINING FINALIZADO")
print(f"⏱️ Tiempo total: {time.time() - start_total:.2f}s")