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
    r'e:\V500\dataset\Original Dataset\Healthy\Healthy162.jpg',
]

for img_path in images:
    if not os.path.exists(img_path):
        continue
    b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    patient_id = os.path.basename(img_path)
    payload = {'image': f'data:image/jpeg;base64,{b64}', 'patientId': patient_id, 'bypassQualityCheck': True, 'patient_history': {}}
    try:
        r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload, timeout=20)
        data = r.json()
        print(f"--- {patient_id} ---")
        print("Quality Score:", data.get('quality_score'))
    except Exception as e:
        print(f"Error on {patient_id}: {e}")
