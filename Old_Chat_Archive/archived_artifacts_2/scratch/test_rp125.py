import requests
import json
import base64
import os

img_path = r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa125.jpg'
b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
patient_id = os.path.basename(img_path)

payload = {
    'image': f'data:image/jpeg;base64,{b64}',
    'patientId': patient_id,
    'bypassQualityCheck': True,
    'patient_history': {}
}

r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload)
data = r.json()
print("Verdict:", data.get('verdict'))
print("AI Confidence:", data.get('ai_confidence'))
print("Quality Score:", data.get('quality_score'))
print("Image Quality dict:", data.get('image_quality'))
