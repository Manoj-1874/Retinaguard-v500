import os
import base64
import json
import requests
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ==============================================================================
# RETINAGUARD V500 - FINAL ARCHITECTURE EVALUATION SCRIPT
# ==============================================================================
# This script sends a dataset of images directly to the running `app.py` Flask 
# server to calculate the real, empirical metrics of the final 10-Expert CDSS.
#
# INSTRUCTIONS:
# 1. Start your backend in a separate terminal: `python app.py`
# 2. Update `DATASET_PATH` below to point to your local validation folder.
#    (The folder should contain 'Healthy' and 'RP' subfolders).
# 3. Run this script. It will generate the final ROC Curve and F1-Scores.
# ==============================================================================

# CONFIGURATION
API_URL = "http://127.0.0.1:5000/analyze"
DATASET_PATH = r"E:\Path\To\Your\Dataset" # UPDATE THIS TO YOUR DATASET FOLDER

def encode_image_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def evaluate_model():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset path '{DATASET_PATH}' not found!")
        print("Please update DATASET_PATH to point to your actual image folder.")
        return

    y_true = []
    y_pred = []
    y_scores = []
    
    classes = ['Healthy', 'RP']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"\n📂 Scanning {class_name} images...")
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # 1. Base64 Encode Image
                b64_img = encode_image_to_base64(img_path)
                
                # 2. Send to app.py API
                payload = {"image": b64_img}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    diagnosis = data.get('diagnosis', '')
                    risk_score = data.get('risk_score', 0) / 100.0
                    
                    # 3. Record Results
                    y_true.append(class_idx)
                    
                    if "HEALTHY" in diagnosis.upper():
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
                        
                    y_scores.append(risk_score)
                    
                    print(f"✅ [SUCCESS] {img_name} -> Predicted: {diagnosis} (Risk: {risk_score*100:.1f}%)")
                else:
                    print(f"❌ [API ERROR] {img_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ [EXCEPTION] {img_name}: {e}")

    # ==========================================
    # CALCULATE METRICS
    # ==========================================
    if not y_true:
        print("\n⚠️ No images were successfully processed. Check your dataset path.")
        return

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, cohen_kappa_score, matthews_corrcoef
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "="*50)
    print("🏆 RETINAGUARD V500 FINAL METRICS (APP.PY)")
    print("="*50)
    print(f"Total Images Evaluated: {len(y_true)}")
    print("-" * 50)
    
    # Base Metrics
    print(f"Overall Accuracy:                 {acc*100:.2f}%")
    print(f"Sensitivity (True Positive Rate): {rec*100:.2f}%")
    
    # Calculate Specificity and NPV
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"Specificity (True Negative Rate): {specificity*100:.2f}%")
        print(f"Positive Predictive Value (PPV):  {prec*100:.2f}%")
        print(f"Negative Predictive Value (NPV):  {npv*100:.2f}%")
    else:
        print("Specificity/NPV: Cannot calculate (missing classes in results)")
    
    # Advanced Statistical Metrics
    print("-" * 50)
    print(f"F1-Score (Harmonic Mean):         {f1:.4f}")
    print(f"Matthews Correlation (MCC):       {mcc:.4f}  (1.0 is perfect)")
    print(f"Cohen's Kappa (Reliability):      {kappa:.4f}  (>0.8 is excellent)")
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"                 Predicted Healthy    Predicted RP")
    if cm.shape == (2,2):
        print(f"Actual Healthy   [{tn}]                  [{fp}]")
        print(f"Actual RP        [{fn}]                  [{tp}]")
    else:
        print(cm)
    
    # ==========================================
    # GENERATE ACADEMIC GRAPHS
    # ==========================================
    try:
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12,5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        # 2. Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_vals, precision_vals)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(r"E:\V500\Research_Paper_Proofs\9_Final_Architecture_Solutions\V500_Final_Clinical_Graphs.png", dpi=300)
        print("\n📈 ROC and Precision-Recall Curves saved to V500_Final_Clinical_Graphs.png")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"PR AUC Score:  {pr_auc:.4f}")
        print("="*50)
    except Exception as e:
        print(f"\n⚠️ Could not generate graphs: {e}")

if __name__ == "__main__":
    print("Starting Final V500 Evaluation against live app.py...")
    evaluate_model()
