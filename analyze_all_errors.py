import requests
import json
import base64
import os

rp_fns = ["Retinitis Pigmentosa100.jpg", "Retinitis Pigmentosa11.jpg", "Retinitis Pigmentosa125.jpg", "Retinitis Pigmentosa126.jpg", "Retinitis Pigmentosa135.jpg", "Retinitis Pigmentosa136.jpg", "Retinitis Pigmentosa16.jpg", "Retinitis Pigmentosa18.jpg", "Retinitis Pigmentosa34.jpg", "Retinitis Pigmentosa35.jpg", "Retinitis Pigmentosa36.jpg", "Retinitis Pigmentosa37.jpg", "Retinitis Pigmentosa47.jpg", "Retinitis Pigmentosa63.jpg", "Retinitis Pigmentosa65.jpg", "Retinitis Pigmentosa66.jpg", "Retinitis Pigmentosa67.jpg", "Retinitis Pigmentosa68.jpg", "Retinitis Pigmentosa89.jpg", "Retinitis Pigmentosa93.jpg"]
healthy_fps = ["Healthy1008.jpg", "Healthy141.jpg", "Healthy148.jpg", "Healthy157.jpg", "Healthy161.jpg", "Healthy162.jpg", "Healthy163.jpg", "Healthy166.jpg", "Healthy176.jpg", "Healthy177.jpg", "Healthy178.jpg", "Healthy182.jpg", "Healthy183.jpg", "Healthy191.jpg", "Healthy197.jpg"]

def analyze_list(images, is_rp):
    results = []
    for img_name in images:
        if not is_rp:
            path = os.path.join(r"e:\V500\Dataset\Original Dataset\Healthy", img_name)
        else:
            path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img_name)
        
        with open(path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
            
        payload = {"image": f"data:image/jpeg;base64,{img_data}", "bypassQualityCheck": True}
        r = requests.post("http://127.0.0.1:5001/api/analyze", json=payload)
        data = r.json()
        
        ai = data.get("ai_confidence", 0)
        
        # Calculate clinical_rp_votes, critical_count, mild_findings manually to see what they are
        clinical_rp_votes = 0
        critical_count = 0
        mild_findings = 0
        has_patho = False
        
        if "expert_opinions" in data:
            for exp in data["expert_opinions"]:
                sev = exp.get("severity", "NORMAL")
                name = exp.get("name", "")
                
                if sev == "CRITICAL": critical_count += 1
                if sev in ["MODERATE", "CRITICAL"] and name != "AI Pattern Recognition":
                    clinical_rp_votes += 1
                if sev == "MILD" and name != "AI Pattern Recognition":
                    mild_findings += 1
                    
                if name in ["Vessel Attenuation (TRIAD #2)", "Bone Spicule Pigmentation (TRIAD #1)"]:
                    if sev in ["MODERATE", "CRITICAL"]:
                        has_patho = True
                        
        results.append(f"{img_name:30} | AI: {ai:4.1f}% | Votes: {clinical_rp_votes} | Crit: {critical_count} | Mild: {mild_findings} | Patho: {has_patho} | Verdict: {data.get('verdict_code')}")
    return results

print("=== FALSE NEGATIVES (True RP) ===")
for r in analyze_list(rp_fns, True):
    print(r)

print("\n=== FALSE POSITIVES (True Healthy) ===")
for r in analyze_list(healthy_fps, False):
    print(r)
