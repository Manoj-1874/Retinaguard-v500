import requests
import json
import base64
import os

images = [
    "Retinitis Pigmentosa100.jpg",
    "Retinitis Pigmentosa35.jpg",
    "Healthy191.jpg",
    "Healthy157.jpg"
]

for img_name in images:
    if img_name.startswith("Healthy"):
        path = os.path.join(r"e:\V500\Dataset\Original Dataset\Healthy", img_name)
    else:
        path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img_name)
    
    with open(path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {"image": f"data:image/jpeg;base64,{img_data}", "bypassQualityCheck": True}
    r = requests.post("http://127.0.0.1:5001/api/analyze", json=payload)
    print(f"\n--- {img_name} ---")
    data = r.json()
    print(json.dumps(data, indent=2))
