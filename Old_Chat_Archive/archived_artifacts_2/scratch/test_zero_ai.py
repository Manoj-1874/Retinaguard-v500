import requests
import json
import base64
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def check_img(img_path):
    b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    patient_id = os.path.basename(img_path)
    payload = {'image': f'data:image/jpeg;base64,{b64}', 'patientId': patient_id, 'bypassQualityCheck': True, 'patient_history': {}}
    try:
        r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get('quality_score', 100) < 10:
                return (patient_id, data.get('ai_confidence'))
    except:
        pass
    return None

def analyze_dir(directory):
    images = glob.glob(os.path.join(directory, '*.jpg'))
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(check_img, images):
            if res:
                results.append(res)
    return results

print("=== RP ZERO QUALITY ===")
rp_res = analyze_dir(r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa')
for p, ai in rp_res:
    print(f"{p}: {ai}")

print("=== HEALTHY ZERO QUALITY ===")
h_res = analyze_dir(r'e:\V500\dataset\Original Dataset\Healthy')
for p, ai in h_res:
    print(f"{p}: {ai}")
