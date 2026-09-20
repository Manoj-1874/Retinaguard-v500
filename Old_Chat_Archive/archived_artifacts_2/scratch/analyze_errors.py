"""Analyze current batch evaluation results to identify error patterns"""
import json
import numpy as np

data = json.load(open('e:/V500/Evaluation_Results/V500_Evaluation_Results.json'))
results = data['per_image_results']

RP_POSITIVE_VERDICTS = {"CLASSIC_RP", "RP_POSITIVE", "RP_SINE_PIGMENTO", 
    "RP_RPA", "RP_SECTORAL", "SUSPICIOUS", "BORDERLINE", "SUSPICIOUS_ISOLATED"}

# Categorize results
fp_images = [r for r in results if r['true_label'] == 'Healthy' and r['verdict'] in RP_POSITIVE_VERDICTS]
fn_images = [r for r in results if r['true_label'] == 'RP' and r['verdict'] not in RP_POSITIVE_VERDICTS]
tp_images = [r for r in results if r['true_label'] == 'RP' and r['verdict'] in RP_POSITIVE_VERDICTS]
tn_images = [r for r in results if r['true_label'] == 'Healthy' and r['verdict'] not in RP_POSITIVE_VERDICTS]

print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(f"\n--- FALSE POSITIVES ({len(fp_images)} healthy images wrongly flagged) ---")
fp_by_verdict = {}
for r in fp_images:
    v = r['verdict']
    fp_by_verdict.setdefault(v, []).append(r)
for v, items in sorted(fp_by_verdict.items()):
    ai_scores = sorted([r['ai_prob'] for r in items])
    print(f"  {v}: {len(items)} images (AI range: {min(ai_scores):.1f}%-{max(ai_scores):.1f}%)")

print(f"\n--- FALSE NEGATIVES ({len(fn_images)} RP images missed) ---")
fn_by_verdict = {}
for r in fn_images:
    v = r['verdict']
    fn_by_verdict.setdefault(v, []).append(r)
for v, items in sorted(fn_by_verdict.items()):
    ai_scores = sorted([r['ai_prob'] for r in items])
    print(f"  {v}: {len(items)} images (AI range: {min(ai_scores):.1f}%-{max(ai_scores):.1f}%)")

# AI-only threshold sweep
print(f"\n{'='*70}")
print("AI-ONLY THRESHOLD SWEEP")
print(f"{'='*70}")

rp_ai = [r['ai_prob'] for r in results if r['true_label'] == 'RP']
h_ai = [r['ai_prob'] for r in results if r['true_label'] == 'Healthy']

best_acc = 0
best_t = 0
for t in np.arange(10, 90, 0.5):
    tp = sum(1 for x in rp_ai if x >= t)
    tn = sum(1 for x in h_ai if x < t)
    fn = len(rp_ai) - tp
    fp = len(h_ai) - tn
    acc = (tp + tn) / (len(rp_ai) + len(h_ai))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    if acc > best_acc:
        best_acc = acc
        best_t = t
        best_m = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec, 'f1': f1}

print(f"\nBest AI-only threshold: {best_t:.1f}%")
print(f"Acc={best_acc*100:.1f}%, Prec={best_m['prec']*100:.1f}%, Rec={best_m['rec']*100:.1f}%, F1={best_m['f1']*100:.1f}%")
print(f"TP={best_m['tp']} TN={best_m['tn']} FP={best_m['fp']} FN={best_m['fn']}")

# Show different thresholds
print(f"\nThreshold sweep:")
for t in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
    tp = sum(1 for x in rp_ai if x >= t)
    tn = sum(1 for x in h_ai if x < t)
    fn = len(rp_ai) - tp
    fp = len(h_ai) - tn
    acc = (tp + tn) / (len(rp_ai) + len(h_ai))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"  T={t}%: Acc={acc*100:.1f}%, Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, F1={f1*100:.1f}% | TP={tp} TN={tn} FP={fp} FN={fn}")

# Analyze overlap zone
print(f"\n{'='*70}")
print("OVERLAP ZONE ANALYSIS (AI 30-60%)")
print(f"{'='*70}")
rp_overlap = [r for r in results if r['true_label'] == 'RP' and 30 <= r['ai_prob'] <= 60]
h_overlap = [r for r in results if r['true_label'] == 'Healthy' and 30 <= r['ai_prob'] <= 60]
print(f"RP images in 30-60% zone: {len(rp_overlap)}")
print(f"Healthy images in 30-60% zone: {len(h_overlap)}")
print(f"Total in overlap zone: {len(rp_overlap) + len(h_overlap)}")
print(f"RP ratio in overlap: {len(rp_overlap)/(len(rp_overlap)+len(h_overlap))*100:.1f}%")

# Verdict distribution for TP
print(f"\n--- TP VERDICT DISTRIBUTION ---")
tp_by_verdict = {}
for r in tp_images:
    v = r['verdict']
    tp_by_verdict.setdefault(v, []).append(r)
for v, items in sorted(tp_by_verdict.items()):
    ai_scores = [r['ai_prob'] for r in items]
    print(f"  {v}: {len(items)} images (AI: {min(ai_scores):.1f}%-{max(ai_scores):.1f}%, mean={np.mean(ai_scores):.1f}%)")

# Key insight
print(f"\n{'='*70}")
print("KEY INSIGHT")
print(f"{'='*70}")
print(f"Current system: Acc={68.84:.1f}%")
print(f"Best AI-only:   Acc={best_acc*100:.1f}% (threshold={best_t:.1f}%)")
print(f"Gap to target:  {94.9 - best_acc*100:.1f}% (needs features to bridge this)")
print(f"Current rules HURT accuracy by {best_acc*100 - 68.84:.1f}%")
