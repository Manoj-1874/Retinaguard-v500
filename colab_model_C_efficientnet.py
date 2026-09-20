# ==============================================================================
# MODEL C: EFFICIENTNET-B4 (STATE-OF-THE-ART ARCHITECTURE)
# ==============================================================================
# This script completely throws away the 10-year-old ResNet50 architecture 
# and replaces it with EfficientNet-B4, which is currently the gold standard 
# in medical imaging.
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
import os

print("="*60)
print(" EFFICIENTNET-B4 (MODEL C) GENERATOR")
print("="*60)

# 1. Build the new Architecture
print("\n[1] Downloading EfficientNet-B4 weights from Google...")
base_model_c = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base for the first phase
base_model_c.trainable = False

# Add our custom medical classification head
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Reuse the augmentation from earlier
x = keras.applications.efficientnet.preprocess_input(x) # EfficientNet specific scaling
x = base_model_c(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_c = keras.Model(inputs, outputs)

# 2. Compile Model
model_c.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\n[2] Training the new classification head (Phase 1)...")
callbacks_c = [
    callbacks.EarlyStopping(monitor='val_auc', patience=4, mode='max', restore_best_weights=True),
]

history_c = model_c.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks_c,
    verbose=1
)

# 3. Fine-Tune EfficientNet
print("\n[3] Unfreezing the top 40 layers of EfficientNet (Phase 2)...")
base_model_c.trainable = True
for layer in base_model_c.layers[:-40]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model_c.compile(
    optimizer=keras.optimizers.Adam(1e-5), # Tiny learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

save_path_c = '/content/drive/MyDrive/efficientnet_model.h5'
callbacks_c_fine = [
    callbacks.EarlyStopping(monitor='val_auc', patience=6, mode='max', restore_best_weights=True),
    callbacks.ModelCheckpoint(save_path_c, monitor='val_auc', mode='max', save_best_only=True)
]

print("\n[4] Fine-Tuning EfficientNet (This is where the magic happens)...")
history_c_fine = model_c.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks_c_fine,
    verbose=1
)

# 4. Final Results
best_val_acc = max(history_c_fine.history.get('val_accuracy', [0]))
best_val_auc = max(history_c_fine.history.get('val_auc', [0]))

print("\n" + "="*60)
print(" MODEL C (EFFICIENTNET) FINAL RESULTS")
print("="*60)
print(f"Accuracy: {best_val_acc*100:.2f}%")
print(f"ROC-AUC:  {best_val_auc:.4f}")
print(f"Saved to: {save_path_c}")
