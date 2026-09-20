#!/usr/bin/env python
"""
Phase 1: Find optimal AI-only threshold from batch output data.
Phase 2: Run full diagnostic to extract rp_score + all features.
Phase 3: Find optimal composite score weights.
"""
import numpy as np
import os, sys, json

# ============================================================
# PHASE 1: AI-only threshold optimization (from batch output)
# ============================================================

# RP AI probabilities (from batch output, excluding 2 API_ERROR cases)
rp_ai = [
    70.0, 85.2, 45.7, 59.7, 44.6, 32.2, 36.2, 39.1, 56.0, 33.4,
    69.2, 71.1, 65.5, 75.4, 78.8, 66.7, 56.4, 58.8, 73.2, 68.5,
    58.6, 63.1, 91.2, 70.5, 64.9, 67.1, 72.3, 69.9, 76.1, 76.1,
    71.5, 90.3, 76.3, 56.2, 47.8, 63.0, 57.1, 56.4, 92.5, 58.4,
    46.7, 47.4, 65.7, 46.2, 67.3, 51.2, 72.6, 56.3, 60.3, 51.3,
    36.1, 38.1, 74.9, 59.6, 59.8, 56.2, 49.2, 67.0, 73.2, 60.5,
    39.3, 66.0, 71.1, 56.6, 61.9, 54.9, 64.2, 41.1, 93.2, 59.0,
    61.3, 60.4, 51.3, 95.5, 82.0, 81.6, 80.3, 95.1, 81.4, 37.9,
    68.1, 55.4, 41.5, 31.8, 45.1, 38.6, 17.4, 18.4, 19.5, 45.1,
    45.3, 43.7, 44.1, 57.4, 42.9, 51.2, 46.5, 51.1, 62.9, 48.1,
    50.7, 89.8, 55.4, 48.0, 47.4, 48.5, 89.0, 45.3, 48.9, 36.5,
    36.9, 35.7, 58.8, 49.5, 46.8, 38.5, 88.4, 53.5, 77.9, 70.5,
    55.9, 86.3, 78.0, 92.4, 85.3, 76.5, 85.6, 38.4, 87.7, 87.7,
    59.1, 47.0, 46.9, 71.2, 39.6
]

healthy_ai = [
    14.6, 29.5, 14.9, 20.5, 22.9, 23.2, 31.1, 20.7, 19.1, 35.6,
    27.0, 55.1, 23.7, 33.8, 25.6, 31.5, 18.3, 16.6, 28.8, 31.6,
    36.9, 22.2, 18.0, 34.6, 13.4, 18.1, 49.9, 35.1, 31.0, 41.9,
    35.2, 55.2, 36.3, 30.4, 50.9, 31.1, 35.0, 48.8, 23.6, 27.8,
    25.0, 48.2, 41.5, 44.7, 27.7, 47.2, 52.7, 34.7, 24.4, 46.3,
    49.9, 50.2, 21.2, 35.9, 34.1, 41.5, 49.7, 36.1, 46.2, 42.9,
    32.6, 28.6, 45.7, 46.7, 56.6, 45.4, 55.8, 48.3, 23.7, 38.2,
    29.4, 47.3, 53.3, 40.3, 41.5, 19.8, 34.2, 28.9, 22.0, 59.3,
    32.5, 32.7, 41.7, 22.4, 37.6, 30.6, 45.3, 38.8, 41.6, 45.8,
    31.2, 34.0, 26.7, 41.6, 71.6, 54.9, 53.6, 31.1, 25.7, 54.2,
    31.6, 48.0, 13.7, 44.6, 35.7, 23.0, 30.8, 30.5, 40.2, 40.0,
    64.8, 53.9, 66.7, 25.5, 40.1, 24.2, 28.2, 55.3, 72.2, 26.6,
    37.6, 23.1, 38.7, 27.4, 67.4, 41.1, 27.1, 49.1, 25.5, 27.8,
    50.3, 33.7, 27.4, 56.7, 39.4, 47.2, 20.8, 44.1, 36.1
]

print("=" * 70)
print("PHASE 1: AI-ONLY THRESHOLD OPTIMIZATION")
print("=" * 70)
print(f"RP images: {len(rp_ai)}, Healthy images: {len(healthy_ai)}")
print(f"Total: {len(rp_ai) + len(healthy_ai)}")
print()

# Find optimal threshold
results = []
for t_10 in range(0, 1000):  # 0.0 to 99.9 in 0.1 steps
    t = t_10 / 10.0
    tp = sum(1 for x in rp_ai if x >= t)
    tn = sum(1 for x in healthy_ai if x < t)
    fn = len(rp_ai) - tp
    fp = len(healthy_ai) - tn
    total = len(rp_ai) + len(healthy_ai)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    results.append((t, acc, prec, recall, f1, spec, tp, tn, fp, fn))

# Sort by accuracy
results.sort(key=lambda x: (-x[1], -x[4]))

print("Top 10 thresholds by ACCURACY:")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Spec':>10} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
print("-" * 95)
for r in results[:10]:
    print(f"{r[0]:>10.1f} {r[1]*100:>10.1f}% {r[2]*100:>10.1f}% {r[3]*100:>10.1f}% {r[4]*100:>10.1f}% {r[5]*100:>10.1f}% {r[6]:>5} {r[7]:>5} {r[8]:>5} {r[9]:>5}")

# Sort by F1
results.sort(key=lambda x: -x[4])
print("\nTop 10 thresholds by F1-SCORE:")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Spec':>10} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
print("-" * 95)
for r in results[:10]:
    print(f"{r[0]:>10.1f} {r[1]*100:>10.1f}% {r[2]*100:>10.1f}% {r[3]*100:>10.1f}% {r[4]*100:>10.1f}% {r[5]*100:>10.1f}% {r[6]:>5} {r[7]:>5} {r[8]:>5} {r[9]:>5}")

print()
print("=" * 70)
print("KEY INSIGHT: Maximum achievable accuracy with AI-ONLY threshold")
print("=" * 70)
best = max(results, key=lambda x: x[1])
print(f"Best threshold: {best[0]}%")
print(f"Best accuracy: {best[1]*100:.1f}%")
print(f"TP={best[6]}, TN={best[7]}, FP={best[8]}, FN={best[9]}")
print(f"Precision={best[2]*100:.1f}%, Recall={best[3]*100:.1f}%, F1={best[4]*100:.1f}%")
print()
print("TARGET: Accuracy=94.9%, Precision=93.0%, Recall=97.1%, F1=95.0%")
print(f"GAP: {(0.949 - best[1])*100:.1f}% accuracy gap to fill with clinical features")
print()

# Distribution analysis
print("=" * 70)
print("DISTRIBUTION ANALYSIS")
print("=" * 70)
for range_start in range(0, 100, 10):
    range_end = range_start + 10
    rp_count = sum(1 for x in rp_ai if range_start <= x < range_end)
    h_count = sum(1 for x in healthy_ai if range_start <= x < range_end)
    print(f"  AI {range_start:>3}-{range_end:>3}%: RP={rp_count:>3}, Healthy={h_count:>3}, Overlap={'HIGH' if rp_count > 0 and h_count > 0 else 'none'}")

# FN analysis at optimal threshold
print()
print("=" * 70)
print(f"FALSE NEGATIVES at threshold {best[0]}% (RP images MISSED):")
print("=" * 70)
fn_list = sorted([x for x in rp_ai if x < best[0]])
for x in fn_list:
    print(f"  AI={x:.1f}%")
print(f"Total FN: {len(fn_list)}")

print()
print("=" * 70)
print(f"FALSE POSITIVES at threshold {best[0]}% (Healthy images wrongly flagged):")
print("=" * 70)
fp_list = sorted([x for x in healthy_ai if x >= best[0]], reverse=True)
for x in fp_list:
    print(f"  AI={x:.1f}%")
print(f"Total FP: {len(fp_list)}")
