import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

print("==================================================")
print(" UNSEEN TEST EVALUATION: RANDOM FOREST (META-LEARNER)")
print("==================================================")

BASE_MODEL_PATH = "e:/V500/models/finetuned_model.h5"
RF_MODEL_PATH = "e:/V500/models/meta_learner.pkl"  # Downloaded from Colab
NEW_RP_DIR = r"e:\V500\Dataset\test_dataset\eye\test\Retinitis Pigmentosa"
NEW_HEALTHY_DIR = r"e:\V500\Dataset\test_dataset\eye\test\Healthy"
IMG_SIZE = (224, 224)

if not os.path.exists(RF_MODEL_PATH):
    print(f"[!] Meta-learner not found at {RF_MODEL_PATH}")
    print("[!] Please download 'meta_learner.pkl' from Google Drive (created by colab_model_B_stacking.py) and place it in the models directory.")
    exit()

print(f"\n[1/4] Loading base CNN (feature extractor)...")
base_model = keras.models.load_model(BASE_MODEL_PATH, compile=False)
feature_extractor = keras.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

unseen_rp = os.listdir(NEW_RP_DIR)
unseen_healthy = os.listdir(NEW_HEALTHY_DIR)

print(f"\n[2/4] Loading and preprocessing test images...")
def load_images(file_list, folder, label):
    images = []
    labels = []
    for fname in file_list:
        path = os.path.join(folder, fname)
        try:
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img_array = keras.utils.img_to_array(img)
            # ResNet50 preprocessing used for the base model
            img_array = keras.applications.resnet_v2.preprocess_input(img_array)
            images.append(img_array)
            labels.append(label)
        except Exception:
            pass
    return images, labels

rp_images, rp_labels = load_images(unseen_rp, NEW_RP_DIR, 1)
healthy_images, healthy_labels = load_images(unseen_healthy, NEW_HEALTHY_DIR, 0)

X_test_img = np.array(rp_images + healthy_images)
y_true = np.array(rp_labels + healthy_labels)

if len(X_test_img) == 0:
    print("[!] No images to test. Exiting.")
    exit()

print(f"\n[3/4] Extracting deep mathematical features (2048-dim) from images...")
X_test_features = feature_extractor.predict(X_test_img, verbose=1)

print(f"\n[4/4] Loading Random Forest model and Predicting...")
rf_model = joblib.load(RF_MODEL_PATH)

y_pred = rf_model.predict(X_test_features)
y_pred_proba = rf_model.predict_proba(X_test_features)[:, 1]

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Healthy", "RP"])

print("\n" + "="*50)
print(" RANDOM FOREST (MODEL B) - UNSEEN DATASET RESULTS")
print("="*50)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC-AUC:  {auc:.4f}")
print("\nConfusion Matrix:")
print(f"True Healthy (TN): {cm[0][0]} | False RP (FP): {cm[0][1]}")
print(f"False Healthy (FN): {cm[1][0]} | True RP (TP): {cm[1][1]}")
print("\nClassification Report:")
print(report)
print("==================================================")
