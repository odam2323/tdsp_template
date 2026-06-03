import sys
import os

# 🔥 CONFIGURAR PATH PRIMERO (ANTES DE IMPORTS)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(ROOT)

print("✅ PATH configurado:", ROOT)

from pathlib import Path
import tensorflow as tf
import time

from brain_tumor_classifier.models.efficient import build_efficientnet

# =====================================================
# ⏱️ INICIO
# =====================================================

start_total = time.time()
print("\n🚀 INICIANDO PIPELINE EFFICIENTNET\n")

# =====================================================
# 📂 RUTAS
# =====================================================

# 🔥 DEFINIR ROOT DEL PROYECTO
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_PATH = DATA_DIR / "Training"
TEST_PATH = DATA_DIR / "Testing"

print(f"📁 Train dir: {TRAIN_PATH}")
print(f"📁 Test dir: {TEST_PATH}")

# =====================================================
# ⚙️ PARÁMETROS
# =====================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 8   # 🔥 más seguro en CPU
EPOCHS = 5

# =====================================================
# 📦 DATA LOADERS
# =====================================================

print("\n📥 Cargando datasets...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Clases detectadas: {class_names}")
print(f"🔢 Número de clases: {num_classes}")

# =====================================================
# 🔄 NORMALIZACIÓN
# =====================================================

print("\n🔄 Aplicando normalización...")

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# =====================================================
# ⚡ PERFORMANCE
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# =====================================================
# 🤖 MODELO
# =====================================================

print("\n🧠 Construyendo modelo EfficientNet...")

model = build_efficientnet(
    input_shape=(128, 128, 3),
    num_classes=num_classes
)

model.summary()

# =====================================================
# 💾 CALLBACKS
# =====================================================

print("\n💾 Configurando callbacks...")

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
# 🚀 ENTRENAMIENTO
# =====================================================

print("\n🚀 Entrenando modelo...\n")

t0 = time.time()

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"\n⏱️ Tiempo entrenamiento: {time.time() - t0:.2f}s")

# =====================================================
# 📊 EVALUACIÓN
# =====================================================

print("\n📊 Evaluando modelo...")

loss, acc = model.evaluate(test_ds)

print(f"\n✅ Accuracy EfficientNet: {acc:.4f}")

# =====================================================
# 🏁 FIN
# =====================================================

model.save("models/efficientnet_savedmodel")

print("\n🏁 PIPELINE FINALIZADO")
print(f"⏱️ Tiempo total: {time.time() - start_total:.2f}s\n")