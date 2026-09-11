"""
================================================================================
RETINAGUARD V500 - COMPLETE HYBRID SYSTEM BATCH EVALUATION
================================================================================
This script sends REAL images through the FULL V500 pipeline:
  Image -> Security -> Camera Calibrator -> CNN -> 10 Experts -> 8 Rules -> VERDICT

It then calculates real, publishable metrics:
  Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Confusion Matrix

IMPORTANT: app.py must be running (python app.py) before executing this script!
================================================================================
"""

import os
import sys
import json
import base64
import time
import requests
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== CONFIGURATION ========================
API_URL = "http://127.0.0.1:5001/api/analyze"

# Dataset paths - CHANGE THESE if your folder names are different
DATASET_ROOT = r"e:\V500\Dataset\Original Dataset"
RP_FOLDER = os.path.join(DATASET_ROOT, "Retinitis Pigmentosa")
HEALTHY_FOLDER = os.path.join(DATASET_ROOT, "Healthy")

# Output paths
OUTPUT_DIR = r"e:\V500\Evaluation_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many images to test (set to None to test ALL images)
MAX_RP_IMAGES = None      # Test ALL 139 RP images
MAX_HEALTHY_IMAGES = 139  # Match with 139 Healthy images (balanced test set)

# RP-Positive verdict codes (these mean "the system thinks it's RP")
RP_POSITIVE_VERDICTS = {
    "CLASSIC_RP", "RP_POSITIVE", "RP_SINE_PIGMENTO", 
    "RP_RPA", "RP_SECTORAL", "SUSPICIOUS", "BORDERLINE"
}

# RP-Negative verdict codes (these mean "the system thinks it's NOT RP")
RP_NEGATIVE_VERDICTS = {
    "HEALTHY", "OTHER_DISEASE"
}

# ===============================================================


def load_image_as_base64(filepath):
    """Read an image file and convert to base64 string for the API."""
    with open(filepath, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    # Add data URI prefix (the API expects this format)
    ext = Path(filepath).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp"}.get(ext.strip('.'), "jpeg")
    return f"data:image/{mime};base64,{b64}"


def send_to_v500(image_b64, patient_id, age=35, ethnicity="Caucasian"):
    """Send a single image to the running V500 Flask API and get the verdict."""
    payload = {
        "image": image_b64,
        "patientId": patient_id,
        "patient_history": {
            "age": age,
            "gender": "Unknown",
            "ethnicity": ethnicity,
            "symptoms": {},
            "familyHistory": False
        }
    }
    
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            return {
                "success": True,
                "verdict_code": result.get("verdict_code", "UNKNOWN"),
                "score": result.get("score", 0),
                "confidence": result.get("confidence", "UNKNOWN"),
                "ai_probability": result.get("ai_confidence", 0),
            }
        elif resp.status_code == 400:
            # Image rejected by quality checks
            result = resp.json()
            return {
                "success": False,
                "verdict_code": "REJECTED",
                "reason": result.get("error", "Quality failure"),
                "quality_score": result.get("quality_score", 0),
            }
        else:
            return {"success": False, "verdict_code": "API_ERROR", "reason": f"HTTP {resp.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "verdict_code": "CONNECTION_ERROR", "reason": "Is app.py running?"}
    except Exception as e:
        return {"success": False, "verdict_code": "ERROR", "reason": str(e)}


def run_batch_evaluation():
    """Main evaluation function."""
    
    print("=" * 70)
    print("  RETINAGUARD V500 - FULL HYBRID SYSTEM BATCH EVALUATION")
    print("=" * 70)
    
    # 1. Verify folders exist
    if not os.path.exists(RP_FOLDER):
        print(f"[ERROR] RP folder not found: {RP_FOLDER}")
        return
    if not os.path.exists(HEALTHY_FOLDER):
        print(f"[ERROR] Healthy folder not found: {HEALTHY_FOLDER}")
        return
    
    # 2. Collect image paths
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    rp_images = [os.path.join(RP_FOLDER, f) for f in os.listdir(RP_FOLDER) 
                 if Path(f).suffix.lower() in valid_extensions]
    healthy_images = [os.path.join(HEALTHY_FOLDER, f) for f in os.listdir(HEALTHY_FOLDER)
                      if Path(f).suffix.lower() in valid_extensions]
    
    # Limit to configured max
    if MAX_RP_IMAGES:
        rp_images = rp_images[:MAX_RP_IMAGES]
    if MAX_HEALTHY_IMAGES:
        healthy_images = healthy_images[:MAX_HEALTHY_IMAGES]
    
    total = len(rp_images) + len(healthy_images)
    print(f"\n[*] Dataset loaded:")
    print(f"    RP images: {len(rp_images)}")
    print(f"    Healthy images: {len(healthy_images)}")
    print(f"    Total: {total}")
    print(f"\n[*] Testing against FULL V500 pipeline (CNN + 10 Experts + 8 Rules)")
    print("-" * 70)
    
    # 3. Check if API is running
    print("[*] Checking API connection...")
    try:
        health = requests.get("http://127.0.0.1:5001/api/health", timeout=5)
        if health.status_code == 200:
            print("[+] API is ONLINE. Starting evaluation...\n")
        else:
            print("[!] API returned unexpected status. Trying anyway...\n")
    except:
        print("[ERROR] Cannot connect to API. Make sure 'python app.py' is running!")
        return
    
    # 4. Run evaluation
    y_true = []          # Ground truth: 1=RP, 0=Healthy
    y_pred = []          # Predicted: 1=RP, 0=Healthy
    y_scores = []        # Raw AI probabilities for ROC curve
    rejected_count = 0
    error_count = 0
    results_log = []
    
    # Process RP images (ground truth = 1)
    print(f"[*] Processing {len(rp_images)} RP images...")
    for i, img_path in enumerate(rp_images):
        filename = os.path.basename(img_path)
        patient_id = f"RP-{i+1:04d}"
        
        try:
            img_b64 = load_image_as_base64(img_path)
        except Exception as e:
            print(f"    [{i+1}/{len(rp_images)}] SKIP {filename}: Cannot read file ({e})")
            error_count += 1
            continue
        
        result = send_to_v500(img_b64, patient_id)
        
        if result["verdict_code"] == "REJECTED":
            print(f"    [{i+1}/{len(rp_images)}] REJECTED {filename}: {result.get('reason', 'Quality failure')} (Score: {result.get('quality_score', '?')})")
            rejected_count += 1
            continue
        elif result["verdict_code"] in ("CONNECTION_ERROR", "API_ERROR", "ERROR"):
            print(f"    [{i+1}/{len(rp_images)}] ERROR {filename}: {result.get('reason', 'Unknown')}")
            error_count += 1
            continue
        
        verdict = result["verdict_code"]
        is_positive = verdict in RP_POSITIVE_VERDICTS
        
        y_true.append(1)  # Ground truth: RP
        y_pred.append(1 if is_positive else 0)
        y_scores.append(result.get("ai_probability", 0.5) / 100.0 if isinstance(result.get("ai_probability", 0), (int, float)) and result.get("ai_probability", 0) > 1 else result.get("ai_probability", 0.5))
        
        status = "TP" if is_positive else "FN"
        print(f"    [{i+1}/{len(rp_images)}] {status} | {filename} -> {verdict} (AI: {result.get('ai_probability', '?')}%)")
        
        results_log.append({
            "file": filename, "true_label": "RP", "verdict": verdict,
            "correct": is_positive, "ai_prob": result.get("ai_probability", 0)
        })
        
        time.sleep(0.3)  # Small delay to not overwhelm the API
    
    print(f"\n[*] Processing {len(healthy_images)} Healthy images...")
    for i, img_path in enumerate(healthy_images):
        filename = os.path.basename(img_path)
        patient_id = f"H-{i+1:04d}"
        
        try:
            img_b64 = load_image_as_base64(img_path)
        except Exception as e:
            print(f"    [{i+1}/{len(healthy_images)}] SKIP {filename}: Cannot read file ({e})")
            error_count += 1
            continue
        
        result = send_to_v500(img_b64, patient_id)
        
        if result["verdict_code"] == "REJECTED":
            print(f"    [{i+1}/{len(healthy_images)}] REJECTED {filename}: {result.get('reason', 'Quality failure')} (Score: {result.get('quality_score', '?')})")
            rejected_count += 1
            continue
        elif result["verdict_code"] in ("CONNECTION_ERROR", "API_ERROR", "ERROR"):
            print(f"    [{i+1}/{len(healthy_images)}] ERROR {filename}: {result.get('reason', 'Unknown')}")
            error_count += 1
            continue
        
        verdict = result["verdict_code"]
        is_positive = verdict in RP_POSITIVE_VERDICTS
        
        y_true.append(0)  # Ground truth: Healthy
        y_pred.append(1 if is_positive else 0)
        y_scores.append(result.get("ai_probability", 0.5) / 100.0 if isinstance(result.get("ai_probability", 0), (int, float)) and result.get("ai_probability", 0) > 1 else result.get("ai_probability", 0.5))
        
        status = "FP" if is_positive else "TN"
        print(f"    [{i+1}/{len(healthy_images)}] {status} | {filename} -> {verdict} (AI: {result.get('ai_probability', '?')}%)")
        
        results_log.append({
            "file": filename, "true_label": "Healthy", "verdict": verdict,
            "correct": not is_positive, "ai_prob": result.get("ai_probability", 0)
        })
        
        time.sleep(0.3)
    
    # 5. Calculate Metrics
    if len(y_true) < 2:
        print("\n[ERROR] Not enough successful predictions to calculate metrics!")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Advanced Metrics
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC Curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    except:
        fpr, tpr, roc_auc = [0, 1], [0, 1], 0.0
    
    # 6. Print Results
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION RESULTS - RETINAGUARD V500 HYBRID CDSS")
    print("=" * 70)
    print(f"\n  Images Tested:    {len(y_true)}")
    print(f"  Images Rejected:  {rejected_count} (failed quality checks)")
    print(f"  Errors:           {error_count}")
    print(f"\n  {'='*50}")
    print(f"  METRICS (Full Hybrid Pipeline)")
    print(f"  {'='*50}")
    print(f"  Accuracy:         {acc*100:.2f}%")
    print(f"  Precision (PPV):  {prec*100:.2f}%")
    print(f"  Recall (Sens.):   {rec*100:.2f}%")
    print(f"  Specificity:      {specificity*100:.2f}%")
    print(f"  NPV:              {npv*100:.2f}%")
    print(f"  F1-Score:         {f1*100:.2f}%")
    print(f"  ROC-AUC:          {roc_auc:.4f}")
    print(f"  Cohen's Kappa:    {kappa:.4f}")
    print(f"  MCC:              {mcc:.4f}")
    print(f"  FPR (Type I):     {fpr_rate*100:.2f}%")
    print(f"  FNR (Type II):    {fnr_rate*100:.2f}%")
    print(f"  FDR:              {fdr*100:.2f}%")
    print(f"  {'='*50}")
    print(f"\n  CONFUSION MATRIX:")
    print(f"  True Positives (TP):   {tp}  (RP correctly detected)")
    print(f"  True Negatives (TN):   {tn}  (Healthy correctly confirmed)")
    print(f"  False Positives (FP):  {fp}  (Healthy wrongly flagged as RP)")
    print(f"  False Negatives (FN):  {fn}  (RP missed)")
    print("=" * 70)
    
    # 7. Generate Classification Report
    report = classification_report(y_true, y_pred, target_names=["Healthy", "RP"], zero_division=0)
    print(f"\n  DETAILED CLASSIFICATION REPORT:")
    print(report)
    
    # 8. Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Healthy", "RP"], yticklabels=["Healthy", "RP"],
                annot_kws={"size": 20})
    plt.title(f'RetinaGuard V500 Confusion Matrix\nAccuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "V500_Confusion_Matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {cm_path}")
    
    # 9. Save ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00bcd4', lw=2, label=f'V500 Hybrid (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - RetinaGuard V500 Hybrid CDSS', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "V500_ROC_Curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {roc_path}")
    
    # 10. Save Full Text Report
    report_path = os.path.join(OUTPUT_DIR, "V500_Evaluation_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RETINAGUARD V500 HYBRID CDSS - BATCH EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Mendeley Eye Disease Image Dataset (External)\n")
        f.write(f"Total Images Tested: {len(y_true)}\n")
        f.write(f"Images Rejected (Quality): {rejected_count}\n\n")
        f.write(f"METRICS:\n")
        f.write(f"  Accuracy:    {acc*100:.2f}%\n")
        f.write(f"  Precision:   {prec*100:.2f}%\n")
        f.write(f"  Recall:      {rec*100:.2f}%\n")
        f.write(f"  Specificity: {specificity*100:.2f}%\n")
        f.write(f"  NPV:         {npv*100:.2f}%\n")
        f.write(f"  F1-Score:    {f1*100:.2f}%\n")
        f.write(f"  ROC-AUC:     {roc_auc:.4f}\n")
        f.write(f"  Cohen Kappa: {kappa:.4f}\n")
        f.write(f"  MCC:         {mcc:.4f}\n")
        f.write(f"  FPR:         {fpr_rate*100:.2f}%\n")
        f.write(f"  FNR:         {fnr_rate*100:.2f}%\n")
        f.write(f"  FDR:         {fdr*100:.2f}%\n\n")
        f.write(f"CONFUSION MATRIX:\n")
        f.write(f"  TP={tp} | FP={fp}\n")
        f.write(f"  FN={fn} | TN={tn}\n\n")
        f.write(f"CLASSIFICATION REPORT:\n")
        f.write(report + "\n\n")
        f.write(f"PER-IMAGE RESULTS:\n")
        f.write("-" * 60 + "\n")
        for r in results_log:
            status = "CORRECT" if r["correct"] else "WRONG"
            f.write(f"  [{status}] {r['file']} | True: {r['true_label']} | Verdict: {r['verdict']} | AI: {r['ai_prob']}%\n")
    
    print(f"  [+] Saved: {report_path}")
    
    # 11. Save JSON results for further analysis
    json_path = os.path.join(OUTPUT_DIR, "V500_Evaluation_Results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "specificity": round(specificity * 100, 2),
                "npv": round(npv * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "roc_auc": round(roc_auc, 4),
                "cohen_kappa": round(kappa, 4),
                "mcc": round(mcc, 4),
                "fpr": round(fpr_rate * 100, 2),
                "fnr": round(fnr_rate * 100, 2),
                "fdr": round(fdr * 100, 2)
            },
            "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            "total_tested": len(y_true),
            "rejected": rejected_count,
            "per_image_results": results_log
        }, f, indent=2)
    print(f"  [+] Saved: {json_path}")
    
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE!")
    print(f"  All results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_batch_evaluation()
