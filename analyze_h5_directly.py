import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Config
MODEL_PATH = r"e:\V500\models\recovered_model.h5"
DATASET_ROOT = r"e:\V500\Dataset\Original Dataset"
RP_FOLDER = os.path.join(DATASET_ROOT, "Retinitis Pigmentosa")
HEALTHY_FOLDER = os.path.join(DATASET_ROOT, "Healthy")
IMG_SIZE = (224, 224)

def load_and_preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    # The Colab training used keras.applications.resnet_v2.preprocess_input
    # which roughly scales pixels to [-1, 1]
    img_array = keras.applications.resnet_v2.preprocess_input(img_array)
    return img_array

def main():
    print(f"Loading Model: {MODEL_PATH}")
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading Images...")
    images = []
    labels = []

    # Load RP (Class 1)
    rp_files = [f for f in os.listdir(RP_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for f in rp_files:
        path = os.path.join(RP_FOLDER, f)
        images.append(load_and_preprocess_image(path))
        labels.append(1)

    # Load Healthy (Class 0)
    healthy_files = [f for f in os.listdir(HEALTHY_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for f in healthy_files:
        path = os.path.join(HEALTHY_FOLDER, f)
        images.append(load_and_preprocess_image(path))
        labels.append(0)

    X = np.array(images)
    y_true = np.array(labels)
    
    print(f"Loaded {len(X)} total images (RP: {len(rp_files)}, Healthy: {len(healthy_files)})")
    print("Running AI Predictions (this may take a minute without a GPU)...")
    
    y_pred_probs = model.predict(X).flatten()
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_probs)
    print("\n" + "="*50)
    print(" RAW CNN MODEL ANALYSIS (No Hybrid Rules)")
    print("="*50)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=["Healthy", "RP"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "RP"], yticklabels=["Healthy", "RP"])
    plt.title(f'Raw CNN Confusion Matrix (AUC: {auc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.savefig('raw_cnn_analysis_cm.png')
    print("\nSaved confusion matrix to: raw_cnn_analysis_cm.png")

if __name__ == "__main__":
    main()
