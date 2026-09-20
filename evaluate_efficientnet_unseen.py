import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

print("==================================================")
print(" UNSEEN TEST EVALUATION: EFFICIENTNET-B4 MODEL")
print("==================================================")

MODEL_PATH = "e:/V500/models/efficientnet_model.weights.h5"
NEW_RP_DIR = r"e:\V500\Dataset\test_dataset\eye\test\Retinitis Pigmentosa"
NEW_HEALTHY_DIR = r"e:\V500\Dataset\test_dataset\eye\test\Healthy"
IMG_SIZE = (224, 224)

unseen_rp = os.listdir(NEW_RP_DIR)
unseen_healthy = os.listdir(NEW_HEALTHY_DIR)

print(f"\n[1/4] Found {len(unseen_rp)} RP and {len(unseen_healthy)} Healthy images in test set.")
print(f"\n[2/4] Loading images...")

def load_images(file_list, folder, label):
    images = []
    labels = []
    for fname in file_list:
        path = os.path.join(folder, fname)
        try:
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img_array = keras.utils.img_to_array(img)
            # EfficientNet uses its own preprocessing
            img_array = keras.applications.efficientnet.preprocess_input(img_array)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            pass
    return images, labels

rp_images, rp_labels = load_images(unseen_rp, NEW_RP_DIR, 1)
healthy_images, healthy_labels = load_images(unseen_healthy, NEW_HEALTHY_DIR, 0)

X_test = np.array(rp_images + healthy_images)
y_true = np.array(rp_labels + healthy_labels)

if len(X_test) == 0:
    print("[!] No images to test. Exiting.")
    exit()

print(f"\n[3/4] Rebuilding Architecture and Loading Weights...")
try:
    try:
        # First try loading it as a complete model
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("    -> Loaded full model successfully.")
    except:
        print("    -> Full model load failed. Rebuilding EfficientNet-B4 architecture...")
        base_model = keras.applications.EfficientNetB4(weights=None, include_top=False, input_shape=(224, 224, 3))
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.load_weights(MODEL_PATH)
        print("    -> Loaded weights successfully.")
except Exception as e:
    print(f"[!] Failed to load model or weights: {e}")
    exit()

print(f"\n[4/4] Predicting...")
y_pred_proba = model.predict(X_test, verbose=1).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Healthy", "RP"])

print("\n" + "="*50)
print(" EFFICIENTNET-B4 - UNSEEN DATASET RESULTS")
print("="*50)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC-AUC:  {auc:.4f}")
print("\nConfusion Matrix:")
print(f"True Healthy (TN): {cm[0][0]} | False RP (FP): {cm[0][1]}")
print(f"False Healthy (FN): {cm[1][0]} | True RP (TP): {cm[1][1]}")
print("\nClassification Report:")
print(report)
print("==================================================")
