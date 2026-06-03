from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, models

def build_efficientnet(input_shape=(128, 128, 3), num_classes=4):

    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False  # Transfer learning (rápido)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model