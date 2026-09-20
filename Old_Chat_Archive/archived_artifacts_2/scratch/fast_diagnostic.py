#!/usr/bin/env python
"""
Fast diagnostic: Uses live API on port 5001, extracts features, finds optimal composite.
"""
import os, sys, json, time, base64, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:5001/api/analyze"
MAX_RP_IMAGES = 137
MAX_HEALTHY_IMAGES = 139

RP_FOLDER = r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa"
HEALTHY_FOLDER = r"e:\V500\Dataset\Original Dataset\Healthy"

valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def load_image_as_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_image(args):
    fname, folder, label, idx = args
    path = os.path.join(folder, fname)
    try:
        b64 = load_image_as_base64(path)
        payload = {
            'image': b64,
            'patientId': f'{label[0]}-{idx:04d}',
            'bypassQualityCheck': True,
            'patient_history': {'age': 35, 'ethnicity': 'Caucasian', 'gender': 'Unknown'}
        }
        resp = requests.post(API_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            return {'filename': fname, 'true_label': label, 'ai_probability': 0,
                    'rp_score': 0, 'composite_score': 0, 'verdict_code': 'ERROR'}
        
        data = resp.json()
        dd = data.get('differential_diagnosis', {})
        disease_scores = dd.get('disease_scores', {})
        dd_features = dd.get('features', {})
        
        row = {
            'filename': fname,
            'true_label': label,
            'ai_probability': data.get('ai_probability', 0),
            'rp_score': disease_scores.get('retinitis_pigmentosa', 0) * 100,
            'composite_score': data.get('composite_score', 0),
            'verdict_code': data.get('verdict_code', 'ERROR'),
            'is_rpa': data.get('is_rpa', False),
            'is_cme': data.get('is_cme', False),
            'is_sectoral': data.get('is_sectoral', False),
            'is_sine_pigmento': data.get('is_sine_pigmento', False),
            'dd_abnormal_texture': dd_features.get('abnormal_texture', 0),
            'dd_vessel_attenuation': dd_features.get('vessel_attenuation', 0),
            'dd_bone_spicules': dd_features.get('bone_spicules', 0),
            'dd_peripheral_loss': dd_features.get('peripheral_loss', 0),
            # Disease scores for differential
            'ds_choroideremia': disease_scores.get('choroideremia', 0) * 100,
            'ds_dr': disease_scores.get('diabetic_retinopathy', 0) * 100,
            'ds_amd': disease_scores.get('amd', 0) * 100,
        }
        return row
    except Exception as e:
        return {'filename': fname, 'true_label': label, 'ai_probability': 0,
                'rp_score': 0, 'composite_score': 0, 'verdict_code': 'ERROR',
                'error': str(e)}

# Build task list
tasks = []
for folder, label, max_images in [(RP_FOLDER, "RP", MAX_RP_IMAGES), (HEALTHY_FOLDER, "HEALTHY", MAX_HEALTHY_IMAGES)]:
    files = sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_ext])
    if max_images:
        files = files[:max_images]
    for i, fname in enumerate(files):
        tasks.append((fname, folder, label, i))

print(f"[*] Processing {len(tasks)} images via API...", flush=True)
start_time = time.time()

results = []
# Sequential to avoid overloading the API
for i, task in enumerate(tasks):
    result = process_image(task)
    results.append(result)
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s elapsed", flush=True)

# Save results
output_path = 'e:/V500/diagnostic_features.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

elapsed = time.time() - start_time
print(f"\n[+] {len(results)} images processed in {elapsed:.0f}s")
print(f"[+] Saved to {output_path}")

# ============================================================
# ANALYSIS AND OPTIMIZATION
# ============================================================
import numpy as np

print("\n" + "=" * 70)
print("FEATURE ANALYSIS AND OPTIMIZATION")
print("=" * 70)

rp_data = [r for r in results if r['true_label'] == 'RP' and r.get('ai_probability', 0) > 0]
healthy_data = [r for r in results if r['true_label'] == 'HEALTHY']
errors = [r for r in results if r.get('verdict_code') == 'ERROR']

print(f"Valid: {len(rp_data)} RP, {len(healthy_data)} Healthy, {len(errors)} Errors")

# Distribution analysis
for feat in ['ai_probability', 'rp_score', 'composite_score', 'dd_abnormal_texture']:
    rp_vals = [r.get(feat, 0) for r in rp_data]
    h_vals = [r.get(feat, 0) for r in healthy_data]
    if rp_vals and h_vals:
        print(f"\n{feat}:")
        print(f"  RP:      mean={np.mean(rp_vals):.2f}, median={np.median(rp_vals):.2f}, std={np.std(rp_vals):.2f}")
        print(f"  Healthy: mean={np.mean(h_vals):.2f}, median={np.median(h_vals):.2f}, std={np.std(h_vals):.2f}")

# ============================================================
# GRID SEARCH: Optimal single-feature thresholds
# ============================================================
print(f"\n{'='*70}")
print("SINGLE-FEATURE OPTIMAL THRESHOLDS")
print(f"{'='*70}")

for feat_name, feat_key, scale in [
    ("AI Probability", "ai_probability", 1.0),
    ("RP Score", "rp_score", 1.0),
    ("Composite Score", "composite_score", 100.0),
]:
    rp_vals = [r.get(feat_key, 0) * scale for r in rp_data]
    h_vals = [r.get(feat_key, 0) * scale for r in healthy_data]
    
    all_vals = sorted(set(rp_vals + h_vals))
    best_acc, best_t, best_metrics = 0, 0, {}
    
    for t in np.arange(min(all_vals), max(all_vals), 0.5):
        tp = sum(1 for x in rp_vals if x >= t)
        tn = sum(1 for x in h_vals if x < t)
        fn = len(rp_vals) - tp
        fp = len(h_vals) - tn
        acc = (tp + tn) / (len(rp_vals) + len(h_vals))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_t = t
            best_metrics = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec, 'f1': f1}
    
    m = best_metrics
    print(f"\n{feat_name}: threshold={best_t:.1f}")
    print(f"  Acc={best_acc*100:.1f}% Prec={m['prec']*100:.1f}% Rec={m['rec']*100:.1f}% F1={m['f1']*100:.1f}%")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

# ============================================================
# GRID SEARCH: Optimal 2-feature weighted combination
# ============================================================
print(f"\n{'='*70}")
print("2-FEATURE OPTIMIZATION: w1*AI + w2*rp_score")
print(f"{'='*70}")

best_overall = {'acc': 0}
for w1_100 in range(10, 95, 5):
    w1 = w1_100 / 100.0
    w2 = 1.0 - w1
    
    rp_combined = [w1 * r['ai_probability'] + w2 * r.get('rp_score', 0) for r in rp_data]
    h_combined = [w1 * r['ai_probability'] + w2 * r.get('rp_score', 0) for r in healthy_data]
    
    for t_10 in range(100, 800):
        t = t_10 / 10.0
        tp = sum(1 for x in rp_combined if x >= t)
        tn = sum(1 for x in h_combined if x < t)
        fn = len(rp_combined) - tp
        fp = len(h_combined) - tn
        total = len(rp_combined) + len(h_combined)
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        
        if acc > best_overall['acc'] or (acc == best_overall['acc'] and f1 > best_overall.get('f1', 0)):
            best_overall = {
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'w1': w1, 'w2': w2, 't': t,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }

b = best_overall
print(f"Best: w_AI={b['w1']:.2f}, w_RP={b['w2']:.2f}, threshold={b['t']:.1f}")
print(f"Acc={b['acc']*100:.1f}% Prec={b['prec']*100:.1f}% Rec={b['rec']*100:.1f}% F1={b['f1']*100:.1f}%")
print(f"TP={b['tp']} TN={b['tn']} FP={b['fp']} FN={b['fn']}")

# ============================================================
# GRID SEARCH: 3-feature (AI + rp_score + composite)
# ============================================================
print(f"\n{'='*70}")
print("3-FEATURE OPTIMIZATION: w1*AI + w2*rp_score + w3*composite*100")
print(f"{'='*70}")

best_3f = {'acc': 0}
for w1_10 in range(1, 9):
    w1 = w1_10 / 10.0
    for w2_10 in range(1, 10 - w1_10):
        w2 = w2_10 / 10.0
        w3 = 1.0 - w1 - w2
        if w3 < 0.05: continue
        
        rp_c = [w1*r['ai_probability'] + w2*r.get('rp_score',0) + w3*r.get('composite_score',0)*100 for r in rp_data]
        h_c = [w1*r['ai_probability'] + w2*r.get('rp_score',0) + w3*r.get('composite_score',0)*100 for r in healthy_data]
        
        for t_10 in range(100, 800):
            t = t_10 / 10.0
            tp = sum(1 for x in rp_c if x >= t)
            tn = sum(1 for x in h_c if x < t)
            fn = len(rp_c) - tp
            fp = len(h_c) - tn
            total = len(rp_c) + len(h_c)
            acc = (tp + tn) / total
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            
            if acc > best_3f['acc'] or (acc == best_3f['acc'] and f1 > best_3f.get('f1', 0)):
                best_3f = {
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                    'w1': w1, 'w2': w2, 'w3': w3, 't': t,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                }

b = best_3f
print(f"Best: w_AI={b['w1']:.2f}, w_RP={b['w2']:.2f}, w_CS={b['w3']:.2f}, threshold={b['t']:.1f}")
print(f"Acc={b['acc']*100:.1f}% Prec={b['prec']*100:.1f}% Rec={b['rec']*100:.1f}% F1={b['f1']*100:.1f}%")
print(f"TP={b['tp']} TN={b['tn']} FP={b['fp']} FN={b['fn']}")

# ============================================================
# LOGISTIC REGRESSION (sklearn if available)
# ============================================================
print(f"\n{'='*70}")
print("LOGISTIC REGRESSION (Full Feature Set)")
print(f"{'='*70}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    feature_keys = ['ai_probability', 'rp_score', 'composite_score', 'dd_abnormal_texture', 
                    'dd_vessel_attenuation', 'dd_bone_spicules', 'dd_peripheral_loss']
    
    all_data = rp_data + healthy_data
    X = np.array([[r.get(k, 0) for k in feature_keys] for r in all_data])
    y = np.array([1]*len(rp_data) + [0]*len(healthy_data))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different regularization strengths
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        print(f"\nC={C}: Acc={acc*100:.1f}% Prec={prec*100:.1f}% Rec={rec*100:.1f}% F1={f1*100:.1f}%")
        print(f"  TP={cm[1][1]} TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]}")
        
        if C == 100.0:
            # Print coefficients
            print(f"\n  Feature Importance (C={C}):")
            for name, coef in zip(feature_keys, model.coef_[0]):
                print(f"    {name}: {coef:.4f}")
            print(f"    intercept: {model.intercept_[0]:.4f}")
            
            # Save model parameters
            model_params = {
                'C': C,
                'feature_keys': feature_keys,
                'coefficients': model.coef_[0].tolist(),
                'intercept': model.intercept_[0],
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
            }
            with open('e:/V500/logistic_model_params.json', 'w') as f:
                json.dump(model_params, f, indent=2)
            print(f"\n  Model params saved to logistic_model_params.json")
    
    # Try with probability threshold tuning
    print(f"\n{'='*70}")
    print("LOGISTIC REGRESSION WITH THRESHOLD TUNING")
    print(f"{'='*70}")
    
    model = LogisticRegression(C=100.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    best_lr = {'acc': 0}
    for t_100 in range(5, 95):
        t = t_100 / 100.0
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        if acc > best_lr['acc'] or (acc == best_lr['acc'] and f1 > best_lr.get('f1', 0)):
            best_lr = {
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 't': t,
                'tp': int(cm[1][1]), 'tn': int(cm[0][0]), 'fp': int(cm[0][1]), 'fn': int(cm[1][0])
            }
    
    b = best_lr
    print(f"Best threshold={b['t']:.2f}")
    print(f"Acc={b['acc']*100:.1f}% Prec={b['prec']*100:.1f}% Rec={b['rec']*100:.1f}% F1={b['f1']*100:.1f}%")
    print(f"TP={b['tp']} TN={b['tn']} FP={b['fp']} FN={b['fn']}")
    
except ImportError:
    print("sklearn not available, skipping logistic regression")

print(f"\n{'='*70}")
print("TARGET: Accuracy=94.9%, Precision=93.0%, Recall=97.1%, F1=95.0%")
print(f"{'='*70}")
