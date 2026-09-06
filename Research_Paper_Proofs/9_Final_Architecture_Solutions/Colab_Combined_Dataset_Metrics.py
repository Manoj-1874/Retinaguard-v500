# ==============================================================================
# COLAB SCRIPT: COMBINED DATASET METRICS EVALUATION
# ==============================================================================
# Copy and paste this entire script into a Google Colab cell.
# It will mount your Drive, combine both datasets, evaluate your .h5 model,
# and output all the advanced academic metrics (F1, MCC, ROC, Kappa).
# ==============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, cohen_kappa_score, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# 1. MOUNT GOOGLE DRIVE
print("🔄 Mounting Google Drive...")
drive.mount('/content/drive')

# 2. CONFIGURATION (Update these paths if needed)
MODEL_PATH = '/content/drive/MyDrive/RetinaGuard_Project/models/recovered_model.h5' # Update to your .h5 path
DATASET_1_PATH = '/content/drive/MyDrive/dataset'
DATASET_2_PATH = '/content/drive/MyDrive/dataset2'

# Combine the directories to search
DATASET_PATHS = [DATASET_1_PATH, DATASET_2_PATH]
CLASSES = ['Healthy', 'RP']

def evaluate_combined_datasets():
    print(f"\n🔄 Loading Model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    y_true = []
    y_pred = []
    y_scores = []

    print("\n🔍 Scanning Datasets...")
    for class_idx, class_name in enumerate(CLASSES):
        for base_path in DATASET_PATHS:
            class_dir = os.path.join(base_path, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️ Warning: Path not found -> {class_dir}")
                continue
                
            print(f"📂 Reading {class_name} images from {base_path}...")
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    # Load and preprocess image for ResNet50
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    
                    x = img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    
                    # Predict
                    pred_prob = model.predict(x, verbose=0)[0][0]
                    
                    y_true.append(class_idx)
                    y_scores.append(pred_prob)
                    
                    if pred_prob > 0.5:
                        y_pred.append(1) # RP
                    else:
                        y_pred.append(0) # Healthy
                        
                except Exception as e:
                    print(f"❌ Error processing {img_name}: {e}")

    # ==========================================
    # CALCULATE ACADEMIC METRICS
    # ==========================================
    if not y_true:
        print("\n❌ No images processed. Check your dataset paths!")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n" + "="*50)
    print("🏆 COLAB RAW MODEL METRICS (COMBINED DATASETS)")
    print("="*50)
    print(f"Total Images Evaluated: {len(y_true)}")
    print("-" * 50)
    
    print(f"Overall Accuracy:                 {acc*100:.2f}%")
    print(f"Sensitivity (True Positive Rate): {rec*100:.2f}%")
    
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"Specificity (True Negative Rate): {specificity*100:.2f}%")
        print(f"Positive Predictive Value (PPV):  {prec*100:.2f}%")
        print(f"Negative Predictive Value (NPV):  {npv*100:.2f}%")
    
    print("-" * 50)
    print(f"F1-Score (Harmonic Mean):         {f1:.4f}")
    print(f"Matthews Correlation (MCC):       {mcc:.4f}")
    print(f"Cohen's Kappa (Reliability):      {kappa:.4f}")
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"                 Predicted Healthy    Predicted RP")
    if cm.shape == (2,2):
        print(f"Actual Healthy   [{tn}]                  [{fp}]")
        print(f"Actual RP        [{fn}]                  [{tp}]")

    # ==========================================
    # GENERATE GRAPHS
    # ==========================================
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    
    plt.figure(figsize=(12,5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_combined_datasets()
