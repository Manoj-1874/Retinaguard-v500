import requests, os, time

time.sleep(3)
API_URL = 'http://127.0.0.1:5001/api/analyze'
DATASET_ROOT = r'e:\V500\Dataset\Original Dataset'

test_images = [
    (r'Retinitis Pigmentosa\Retinitis Pigmentosa122.jpg', 'RP-122 (FN)'),
    (r'Retinitis Pigmentosa\Retinitis Pigmentosa14.jpg', 'RP-14 (FN)'),
    (r'Retinitis Pigmentosa\Retinitis Pigmentosa200.jpg', 'RP-200 (FN)'),
    (r'Retinitis Pigmentosa\Retinitis Pigmentosa202.jpg', 'RP-202 (FN)'),
    (r'Healthy\Healthy1008.jpg', 'Healthy-1008 (FP)'),
    (r'Healthy\Healthy104.jpg', 'Healthy-104 (FP)'),
    (r'Healthy\Healthy161.jpg', 'Healthy-161 (FP)'),
]

for rel_path, label in test_images:
    full_path = os.path.join(DATASET_ROOT, rel_path)
    with open(full_path, 'rb') as f:
        try:
            resp = requests.post(API_URL, files={'image': f})
            if resp.status_code == 200:
                data = resp.json()
                print(f'--- {label} ---')
                print('Verdict:', data.get('verdict'))
                print('AI Conf:', f"{data.get('ai_confidence', 0)*100:.1f}%")
                diff = data.get('differential_diagnosis', {})
                print('RP Score:', f"{diff.get('Retinitis Pigmentosa', 0):.1f}%")
                print('Mild Findings:', data.get('clinical_metrics', {}).get('mild_findings', 0))
                print('Clinical Votes:', data.get('clinical_metrics', {}).get('clinical_votes', 0))
            else:
                print(f'{label} failed: {resp.status_code}')
        except Exception as e:
            print(f'{label} failed: {e}')
