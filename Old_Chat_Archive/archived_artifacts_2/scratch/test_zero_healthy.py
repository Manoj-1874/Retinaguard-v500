import requests
import json
import base64
import os
import glob

healthy_dir = r'e:\V500\dataset\Original Dataset\Healthy'
healthy_images = glob.glob(os.path.join(healthy_dir, '*.jpg'))

zero_quality_healthy = []

for img_path in healthy_images:
    b64 = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    patient_id = os.path.basename(img_path)
    payload = {'image': f'data:image/jpeg;base64,{b64}', 'patientId': patient_id, 'bypassQualityCheck': True, 'patient_history': {}}
    try:
        r = requests.post('http://127.0.0.1:5001/api/analyze', json=payload, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get('quality_score', 100) == 0.0:
                zero_quality_healthy.append(patient_id)
    except:
        pass

print("Healthy images with Quality Score 0.0:")
for f in zero_quality_healthy:
    print(f)
print(f"Total: {len(zero_quality_healthy)}")
