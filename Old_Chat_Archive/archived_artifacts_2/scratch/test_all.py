import requests
import json
import base64
import os

images = [
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa12.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa30.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa135.jpg',
    r'e:\V500\dataset\Original Dataset\Healthy\Healthy162.jpg',
    r'e:\V500\dataset\Original Dataset\Healthy\Healthy183.jpg',
]

for img_path in images:
    if not os.path.exists(img_path):
        continue
    b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    patient_id = os.path.basename(img_path)
    payload = {'image': f'data:image/jpeg;base64,{b64}', 'patientId': patient_id, 'bypassQualityCheck': True, 'patient_history': {}}
    r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload)
    data = r.json()

    print(f"--- {patient_id} ---")
    print("Verdict:", data.get('verdict'))
    print("AI Confidence:", data.get('ai_confidence'))
    diff = data.get('differential_diagnosis', {})
    print(f"Top Disease: {diff.get('top_diagnosis')} ({diff.get('top_confidence')}%)")
    print(f"RP Score: {diff.get('disease_scores', {}).get('retinitis_pigmentosa')}")
    print("\n")
