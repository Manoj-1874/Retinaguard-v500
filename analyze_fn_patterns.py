import requests
import base64
import os
import json

rp_fns = ["Retinitis Pigmentosa100.jpg", "Retinitis Pigmentosa11.jpg", "Retinitis Pigmentosa125.jpg", "Retinitis Pigmentosa126.jpg", "Retinitis Pigmentosa135.jpg", "Retinitis Pigmentosa136.jpg", "Retinitis Pigmentosa16.jpg", "Retinitis Pigmentosa18.jpg", "Retinitis Pigmentosa34.jpg", "Retinitis Pigmentosa35.jpg", "Retinitis Pigmentosa36.jpg", "Retinitis Pigmentosa37.jpg", "Retinitis Pigmentosa47.jpg", "Retinitis Pigmentosa63.jpg", "Retinitis Pigmentosa65.jpg", "Retinitis Pigmentosa66.jpg", "Retinitis Pigmentosa67.jpg", "Retinitis Pigmentosa68.jpg", "Retinitis Pigmentosa89.jpg", "Retinitis Pigmentosa93.jpg"]

print("Analyzing FNs to find a unique pattern...")
for img_name in rp_fns:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img_name)
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
    print(f"Top Disease: {diff.get('top_diagnosis')} ({diff.get('top_confidence')}%)")
    
    # Print non-zero features
    active_features = [f"{k}={v}" for k, v in features.items() if v > 0]
    print("Features:", ", ".join(active_features))
