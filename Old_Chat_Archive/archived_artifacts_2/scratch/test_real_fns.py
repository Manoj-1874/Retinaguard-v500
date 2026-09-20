import requests
import base64

fns = ['Retinitis Pigmentosa100.jpg', 'Retinitis Pigmentosa11.jpg', 'Retinitis Pigmentosa16.jpg', 'Retinitis Pigmentosa34.jpg', 'Retinitis Pigmentosa35.jpg', 'Retinitis Pigmentosa36.jpg', 'Retinitis Pigmentosa47.jpg', 'Retinitis Pigmentosa65.jpg', 'Retinitis Pigmentosa67.jpg', 'Retinitis Pigmentosa68.jpg']

for fn in fns:
    path = f'E:/V500/dataset/Original Dataset/Retinitis Pigmentosa/{fn}'
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        r = requests.post('http://127.0.0.1:5001/api/analyze', json={'image_data': b64, 'patient_id': fn, 'bypassQualityCheck': True})
        res = r.json()
        print(f'--- {fn} ---')
        print(f'Verdict: {res.get("verdict_code", "")}')
        print(f'AI Confidence: {res.get("ai_probability", 0)}')
        diff = res.get("differential_diagnosis", {})
        rp_score = diff.get("disease_scores", {}).get("retinitis_pigmentosa", 0) if diff else 0
        print(f'RP Score: {rp_score}')
    except Exception as e:
        print(f'Failed {fn}: {e}')
