import requests
import json
import base64
import os

images = [
    "Retinitis Pigmentosa134.jpg",
    "Retinitis Pigmentosa17.jpg",
    "Retinitis Pigmentosa37.jpg",
    "Retinitis Pigmentosa4.jpg",
    "Retinitis Pigmentosa45.jpg",
    "Retinitis Pigmentosa5.jpg",
    "Retinitis Pigmentosa6.jpg",
    "Retinitis Pigmentosa63.jpg",
    "Retinitis Pigmentosa64.jpg",
    "Retinitis Pigmentosa7.jpg",
    "Retinitis Pigmentosa89.jpg",
]

def analyze_fns():
    dataset_root = r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa"
    for img in images:
        path = os.path.join(dataset_root, img)
        if not os.path.exists(path):
            continue
            
        with open(path, "rb") as f:
            enc = base64.b64encode(f.read()).decode("utf-8")
            
        data = {
            "image": enc,
            "patient_data": {"age": 35, "ethnicity": "Caucasian"},
            "bypassQualityCheck": True
        }
        
        try:
            r = requests.post("http://localhost:5001/api/analyze", json=data)
            if r.status_code != 200:
                print(f"[{img}] Failed with status {r.status_code}: {r.text[:200]}")
                continue
            res = r.json()
            
            ai_conf = res['ai_analysis']['rp_probability']
            verdict = res['decision']['verdict']
            diag = res['differential_diagnosis']['top_diagnosis']
            diag_conf = res['differential_diagnosis']['top_confidence']
            mild = res['clinical_results']['mild_findings_count']
            crit = sum(1 for v in res['clinical_results']['expert_panel'].values() if v['severity'] == 'CRITICAL')
            vessels = res['clinical_results']['expert_panel']['vessel_attenuation']['severity']
            pigment = res['clinical_results']['expert_panel']['bone_spicule']['severity']
            texture = res['clinical_results']['expert_panel']['texture_degeneration']['abnormal_ratio']
            
            print(f"[{img}] AI: {ai_conf*100:.1f}%")
            print(f"  Verdict: {verdict}")
            print(f"  Top Diff: {diag} ({diag_conf:.1f}%)")
            print(f"  Mild: {mild}, Crit: {crit}, Vessels: {vessels}, Pigment: {pigment}, Texture: {texture:.2f}")
            print("-" * 40)
        except Exception as e:
            print(f"Failed {img}: {e}")

if __name__ == "__main__":
    analyze_fns()
