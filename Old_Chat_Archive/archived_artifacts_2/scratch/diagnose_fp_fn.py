# Diagnostic script: Analyze ALL 276 images through the API and log which rules fire
# This gives us the exact distribution of FP causes

import requests
import base64
import os
import json
from collections import defaultdict

DATASET_ROOT = r"E:\V500\dataset\Original Dataset"
RP_FOLDER = os.path.join(DATASET_ROOT, "Retinitis Pigmentosa")
HEALTHY_FOLDER = os.path.join(DATASET_ROOT, "Healthy")
API_URL = "http://127.0.0.1:5001/api/analyze"

valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def analyze_image(path, patient_id):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    r = requests.post(API_URL, json={
        'image_data': b64, 
        'patient_id': patient_id,
        'bypassQualityCheck': True,
        'patient_history': {'age': 35, 'ethnicity': 'Caucasian'}
    }, timeout=30)
    return r.json()

# Collect stats
fp_verdicts = defaultdict(int)  # verdict_code -> count
fp_ai_confidences = []
fn_details = []
fp_details = []

print("="*70)
print("  DIAGNOSTIC: Analyzing FP/FN distribution")
print("="*70)

# First, just analyze healthy images to understand FPs
healthy_files = sorted([f for f in os.listdir(HEALTHY_FOLDER) if os.path.splitext(f)[1].lower() in valid_ext])
print(f"\n[*] Processing {len(healthy_files)} Healthy images...")

tp, tn, fp, fn = 0, 0, 0, 0

RP_POSITIVE_VERDICTS = {"CLASSIC_RP", "RP_POSITIVE", "RP_SINE_PIGMENTO", "RP_RPA", "RP_SECTORAL", "SUSPICIOUS", "BORDERLINE", "SUSPICIOUS_ISOLATED"}

for i, fname in enumerate(healthy_files):
    path = os.path.join(HEALTHY_FOLDER, fname)
    try:
        res = analyze_image(path, f"H-{i+1:04d}")
        vc = res.get('verdict_code', 'ERROR')
        ai_prob = res.get('ai_probability', 0)
        is_positive = vc in RP_POSITIVE_VERDICTS
        
        if is_positive:
            fp += 1
            fp_verdicts[vc] += 1
            fp_ai_confidences.append(ai_prob)
            fp_details.append({
                'file': fname, 
                'verdict_code': vc, 
                'ai_prob': ai_prob,
                'verdict': res.get('verdict', '')[:80]
            })
            print(f"  [{i+1}/{len(healthy_files)}] FP | {fname} -> {vc} (AI: {ai_prob:.1f}%)")
        else:
            tn += 1
    except Exception as e:
        print(f"  [{i+1}/{len(healthy_files)}] ERR | {fname}: {e}")

print(f"\n[*] Processing RP images...")
rp_files = sorted([f for f in os.listdir(RP_FOLDER) if os.path.splitext(f)[1].lower() in valid_ext])

for i, fname in enumerate(rp_files):
    path = os.path.join(RP_FOLDER, fname)
    try:
        res = analyze_image(path, f"RP-{i+1:04d}")
        vc = res.get('verdict_code', 'ERROR')
        ai_prob = res.get('ai_probability', 0)
        is_positive = vc in RP_POSITIVE_VERDICTS
        
        if is_positive:
            tp += 1
        else:
            fn += 1
            fn_details.append({
                'file': fname, 
                'verdict_code': vc, 
                'ai_prob': ai_prob,
                'verdict': res.get('verdict', '')[:80]
            })
            print(f"  [{i+1}/{len(rp_files)}] FN | {fname} -> {vc} (AI: {ai_prob:.1f}%)")
    except Exception as e:
        print(f"  [{i+1}/{len(rp_files)}] ERR | {fname}: {e}")

print("\n" + "="*70)
print("  DIAGNOSTIC RESULTS")
print("="*70)
print(f"\n  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"  Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.1f}%")
print(f"  Precision: {tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "  Precision: N/A")
print(f"  Recall: {tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "  Recall: N/A")
print(f"  Specificity: {tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "  Specificity: N/A")
print(f"  F1: {2*tp/(2*tp+fp+fn)*100:.1f}%" if (2*tp+fp+fn) > 0 else "  F1: N/A")

print(f"\n  FP Breakdown by verdict_code:")
for vc, count in sorted(fp_verdicts.items(), key=lambda x: -x[1]):
    print(f"    {vc}: {count}")

if fp_ai_confidences:
    import statistics
    print(f"\n  FP AI confidence stats:")
    print(f"    Mean: {statistics.mean(fp_ai_confidences):.1f}%")
    print(f"    Median: {statistics.median(fp_ai_confidences):.1f}%")
    print(f"    Min: {min(fp_ai_confidences):.1f}%")
    print(f"    Max: {max(fp_ai_confidences):.1f}%")

print(f"\n  False Negatives ({len(fn_details)}):")
for d in fn_details:
    print(f"    {d['file']}: {d['verdict_code']} (AI: {d['ai_prob']:.1f}%)")

print(f"\n  False Positives (first 20 of {len(fp_details)}):")
for d in fp_details[:20]:
    print(f"    {d['file']}: {d['verdict_code']} (AI: {d['ai_prob']:.1f}%) - {d['verdict']}")

# Save full results
with open(r'E:\V500\Evaluation_Results\diagnostic_fp_fn.json', 'w') as f:
    json.dump({'fp_details': fp_details, 'fn_details': fn_details, 'fp_by_verdict': dict(fp_verdicts)}, f, indent=2)
print(f"\n  [+] Full results saved to E:\\V500\\Evaluation_Results\\diagnostic_fp_fn.json")
