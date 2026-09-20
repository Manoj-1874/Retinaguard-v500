import os
import subprocess
import time
import urllib.request
import sys

print("Starting app.py...", flush=True)
proc = subprocess.Popen(["python", "app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Waiting for server...", flush=True)
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:5001/api/health")
        break
    except:
        time.sleep(1)
else:
    print("Server failed to start!")
    proc.kill()
    sys.exit(1)

print("Server is up! Running batch evaluate...", flush=True)
result = subprocess.run(["python", "batch_evaluate_v500.py"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

print("Killing server...", flush=True)
proc.kill()
