import requests
import json
import base64

img_path = r'e:\V500\dataset\Original Dataset\Retinitis Pigmentosa\Retinitis Pigmentosa12.jpg'
b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
payload = {'image': f'data:image/jpeg;base64,{b64}', 'patientId': 'RP12.jpg', 'bypassQualityCheck': True, 'patient_history': {}}
r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload)
data = r.json()

print("Verdict:", data.get('verdict'))
print("Verdict Code:", data.get('verdict_code'))
print("AI Confidence:", data.get('ai_confidence'))

for v in data.get('expert_opinions', []):
    print(f"{v['name']}: {v['severity']} ({v['detail']})")
