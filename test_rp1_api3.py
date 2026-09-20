import os
import time
import requests
import base64
import subprocess
import json

print("Starting API in background...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open("api_output.log", "w") as out_f:
    proc = subprocess.Popen(["python", "app.py"], stdout=out_f, stderr=subprocess.STDOUT, env=env)

print("Waiting for API to start...")
for _ in range(30):
    try:
        r = requests.get('http://127.0.0.1:5001/api/health')
        if r.status_code == 200:
            break
    except:
        pass
    time.sleep(2)

print("API started! Testing RP1.jpg...")
path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", "Retinitis Pigmentosa1.jpg")
with open(path, "rb") as f:
    b = base64.b64encode(f.read()).decode("utf-8")

try:
    resp = requests.post('http://127.0.0.1:5001/api/analyze', json={"image": f"data:image/jpeg;base64,{b}", "bypassQualityCheck": True})
    print(resp.status_code)
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print("Request failed:", e)

print("Killing API...")
proc.kill()
