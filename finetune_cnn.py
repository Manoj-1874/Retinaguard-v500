#!/usr/bin/env python
"""
RetinaGuard V500 — CNN Fine-Tuning Script
==========================================
Unfreezes the last layers of the ResNet50V2 backbone and fine-tunes
on the full dataset (583 RP + 1024 Healthy) with data augmentation.

Expected improvement: From ROC-AUC 0.8445 → 0.95+ (Accuracy ~90-95%)

Usage:
    python finetune_cnn.py

Output:
    models/finetuned_model.h5   — The fine-tuned model
    models/recovered_model.h5   — Backed up original model
"""

import os
import sys
import numpy as np
import shutil

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "e:/V500/models/recovered_model.h5"
OUTPUT_PATH = "e:/V500/models/finetuned_model.h5"
BACKUP_PATH = "e:/V500/models/recovered_model_backup.h5"

RP_FOLDER = r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa"
HEALTHY_FOLDER = r"e:\V500\Dataset\Original Dataset\Healthy"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 15     # Phase 1: Train new head with frozen backbone
EPOCHS_PHASE2 = 30     # Phase 2: Fine-tune last 30 layers
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5  # Very low LR for fine-tuning
VALIDATION_SPLIT = 0.20
UNFREEZE_LAYERS = 30   # Number of layers to unfreeze from the end

# ============================================================
# SETUP
# ============================================================
print("=" * 70)
print("  RETINAGUARD V500 — CNN FINE-TUNING")
print("=" * 70)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

# Verify GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[+] GPU available: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[!] No GPU detected — training will be slow on CPU")
    print("    Consider using Google Colab or a cloud GPU")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n[1/5] Loading dataset...")

valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() not in valid_ext:
            continue
        path = os.path.join(folder, fname)
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        img_array = keras.utils.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)
    return images, labels

rp_images, rp_labels = load_images_from_folder(RP_FOLDER, 1)
healthy_images, healthy_labels = load_images_from_folder(HEALTHY_FOLDER, 0)

X = np.array(rp_images + healthy_images)
y = np.array(rp_labels + healthy_labels)

print(f"  Total images: {len(X)}")
print(f"  RP: {sum(y == 1)}, Healthy: {sum(y == 0)}")

# Train/val split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
)

print(f"  Training: {len(X_train)} ({sum(y_train == 1)} RP, {sum(y_train == 0)} Healthy)")
print(f"  Validation: {len(X_val)} ({sum(y_val == 1)} RP, {sum(y_val == 0)} Healthy)")

# Class weights for imbalanced dataset
n_rp = sum(y_train == 1)
n_healthy = sum(y_train == 0)
total = len(y_train)
class_weights = {
    0: total / (2 * n_healthy),
    1: total / (2 * n_rp)
}
print(f"  Class weights: Healthy={class_weights[0]:.2f}, RP={class_weights[1]:.2f}")

# ============================================================
# DATA AUGMENTATION
# ============================================================
print("\n[2/5] Setting up data augmentation...")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# ============================================================
# LOAD & MODIFY MODEL
# ============================================================
print("\n[3/5] Loading model and preparing for fine-tuning...")

# Backup original model
if not os.path.exists(BACKUP_PATH):
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"  [+] Original model backed up to {BACKUP_PATH}")

# Load the existing model
model = keras.models.load_model(MODEL_PATH, compile=False)

# Find the base model (ResNet50V2)
base_model = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 10:
        base_model = layer
        break

if base_model is None:
    print("  [!] Could not find nested base model, treating entire model as base")
    base_model = model

print(f"  Base model: {base_model.name}")
print(f"  Total layers: {len(base_model.layers)}")
print(f"  Currently trainable params: {sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)}")

# ============================================================
# PHASE 1: Re-train the classification head (frozen backbone)
# ============================================================
print(f"\n[4/5] PHASE 1: Training classification head ({EPOCHS_PHASE1} epochs)...")

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

phase1_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
train_dataset = train_dataset.map(
    lambda x, y_label: (data_augmentation(x, training=True), y_label),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=phase1_callbacks,
    verbose=1
)

phase1_val_acc = max(history1.history.get('val_accuracy', [0]))
phase1_val_auc = max(history1.history.get('val_auc', [0]))
print(f"\n  Phase 1 Results: Val Accuracy={phase1_val_acc*100:.1f}%, Val AUC={phase1_val_auc:.4f}")

# ============================================================
# PHASE 2: Fine-tune the backbone (unfreeze last N layers)
# ============================================================
print(f"\n[5/5] PHASE 2: Fine-tuning last {UNFREEZE_LAYERS} backbone layers ({EPOCHS_PHASE2} epochs)...")

for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

trainable_count = sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
print(f"  Trainable params after unfreezing: {trainable_count:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

phase2_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    callbacks.ModelCheckpoint(OUTPUT_PATH, monitor='val_auc', mode='max', save_best_only=True),
]

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=phase2_callbacks,
    verbose=1
)

phase2_val_acc = max(history2.history.get('val_accuracy', [0]))
phase2_val_auc = max(history2.history.get('val_auc', [0]))
print(f"\n  Phase 2 Results: Val Accuracy={phase2_val_acc*100:.1f}%, Val AUC={phase2_val_auc:.4f}")

# ============================================================
# EVALUATE ON VALIDATION SET
# ============================================================
print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

best_model = keras.models.load_model(OUTPUT_PATH, compile=False)
y_pred_proba = best_model.predict(X_val, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)
cm = confusion_matrix(y_val, y_pred)

print(f"\n  Validation Metrics (Fine-tuned Model):")
print(f"  Accuracy:  {acc*100:.1f}%")
print(f"  Precision: {prec*100:.1f}%")
print(f"  Recall:    {rec*100:.1f}%")
print(f"  F1-Score:  {f1*100:.1f}%")
print(f"  ROC-AUC:   {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

# ============================================================
# DEPLOY
# ============================================================
print(f"\n{'='*70}")
print("DEPLOYMENT")
print(f"{'='*70}")

if auc > 0.88:
    shutil.copy2(OUTPUT_PATH, MODEL_PATH)
    print(f"  [+] Fine-tuned model deployed to {MODEL_PATH}")
    print(f"  [+] Original model preserved at {BACKUP_PATH}")
    print(f"\n  To revert: copy {BACKUP_PATH} -> {MODEL_PATH}")
else:
    print(f"  [!] AUC {auc:.4f} not sufficient. Model NOT deployed.")
    print(f"  [!] Fine-tuned model saved at {OUTPUT_PATH} for review.")

print(f"\n{'='*70}")
print("  DONE! Restart the Flask API (python app.py) and re-run evaluation.")
print(f"{'='*70}")
