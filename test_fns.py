import os, base64, requests, json

url = 'http://127.0.0.1:5001/api/analyze'
images = ['Retinitis Pigmentosa13.jpg', 'Retinitis Pigmentosa100.jpg']

for img in images:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img)
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(url, json={"image": f"data:image/jpeg;base64,{b}", "bypassQualityCheck": True})
    
    try:
        d = json.loads(resp.text)
        if isinstance(d, str): d = json.loads(d)
        print(f"\n--- {img} ---")
        print(f"Verdict: {d.get('diagnosis', {}).get('verdict_code')} | AI: {d.get('diagnosis', {}).get('ai_confidence', 0)}%")
        print(d.get('clinical_features_debug'))
    except Exception as e:
        print(f"Error on {img}: {e}")
