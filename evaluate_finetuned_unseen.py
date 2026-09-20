import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print(" UNSEEN TEST EVALUATION: FINETUNED CNN MODEL")
print("==================================================")

# Paths
MODEL_PATH = "e:/V500/models/finetuned_model.h5"

NEW_RP_DIR = r"E:\V500\unseen_test_data\RP"
NEW_HEALTHY_DIR = r"E:\V500\unseen_test_data\Healthy"

IMG_SIZE = (224, 224)

unseen_rp = os.listdir(NEW_RP_DIR)
unseen_healthy = os.listdir(NEW_HEALTHY_DIR)

print(f"\n[1/2] Found {len(unseen_rp)} RP and {len(unseen_healthy)} Healthy images in test set.")

# Optional: Limit to 100 images each to keep testing balanced and fast (uncomment if needed)
# MAX_TEST = 100
# unseen_rp = unseen_rp[:MAX_TEST]
# unseen_healthy = unseen_healthy[:MAX_TEST]

print(f"\n[2/2] Loading {len(unseen_rp)} RP and {len(unseen_healthy)} Healthy images...")

def load_images(file_list, folder, label):
    images = []
    labels = []
    for fname in file_list:
        path = os.path.join(folder, fname)
        try:
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img_array = keras.utils.img_to_array(img) / 255.0
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

print(f"\n[4/4] Loading Model and Predicting...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"[!] Failed to load model: {e}")
    exit()

y_pred_proba = model.predict(X_test, verbose=1).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Healthy", "RP"])

print("\n" + "="*50)
print(" FINETUNED CNN - UNSEEN DATASET RESULTS")
print("="*50)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC-AUC:  {auc:.4f}")
print("\nConfusion Matrix:")
print(f"True Healthy (TN): {cm[0][0]} | False RP (FP): {cm[0][1]}")
print(f"False Healthy (FN): {cm[1][0]} | True RP (TP): {cm[1][1]}")
print("\nClassification Report:")
print(report)
print("==================================================")
