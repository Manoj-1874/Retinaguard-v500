import base64
import json
import urllib.request
import ssl

ssl_context = ssl._create_unverified_context()
api_url = "http://localhost:5001/api/progression-compare"

# Load Sectoral_RP_Test.jpg as baseline, Fluorescein_Angiography_Test.jpg as current (just for testing API call)
try:
    with open(r"e:\V500\uploads\Sectoral_RP_Test.jpg", "rb") as f:
        baseline_base64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
        
    with open(r"e:\V500\uploads\Fluorescein_Angiography_Test.jpg", "rb") as f:
        current_base64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
        
    payload = {
        "baseline_image": baseline_base64,
        "current_image": current_base64,
        "months_between": 12,
        "baseline_date": "2025-01-15",
        "current_date": "2026-01-15"
    }
    
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    print("Sending request to /api/progression-compare...")
    with urllib.request.urlopen(req, context=ssl_context) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        print("Success! Response:")
        print(json.dumps(res_data, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(e.read().decode('utf-8'))
except Exception as e:
    print("Failed:", e)
