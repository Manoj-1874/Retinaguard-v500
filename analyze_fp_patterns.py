import requests
import base64
import os
import json

healthy_fps = ["Healthy1008.jpg", "Healthy141.jpg", "Healthy148.jpg", "Healthy157.jpg", "Healthy161.jpg", "Healthy162.jpg", "Healthy163.jpg", "Healthy166.jpg", "Healthy176.jpg", "Healthy177.jpg", "Healthy178.jpg", "Healthy182.jpg", "Healthy183.jpg", "Healthy191.jpg", "Healthy197.jpg"]

print("Analyzing FPs to find their texture...")
for img_name in healthy_fps:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Healthy", img_name)
    with open(path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {"image": f"data:image/jpeg;base64,{img_data}", "bypassQualityCheck": True}
    r = requests.post("http://127.0.0.1:5001/api/analyze", json=payload)
    if r.status_code != 200: continue
    data = r.json()
    
    ai = data.get("ai_confidence", 0)
    diff = data.get("differential_diagnosis", {})
    features = diff.get("features", {})
    
    print(f"\n--- {img_name} (AI: {ai}%) ---")
    active_features = [f"{k}={v}" for k, v in features.items() if v > 0]
    print("Features:", ", ".join(active_features))
