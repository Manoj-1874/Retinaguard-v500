import requests
import json
import base64
import os

images = [
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa125.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa134.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa17.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa18.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa40.jpg',
    r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa45.jpg',
    r'e:\V500\dataset\Original Dataset\Healthy\Healthy148.jpg',
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
    print(f"RP Score: {diff.get('disease_scores', {}).get('retinitis_pigmentosa')}")
    features = diff.get('features', {})
    print(f"Abnormal Texture: {features.get('abnormal_texture')}")
    print(f"Vessel Attenuation: {features.get('vessel_attenuation')}")
    print(f"Peripheral Loss: {features.get('peripheral_loss')}")
