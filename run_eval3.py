import os
import subprocess
import time

print("Starting app.py...", flush=True)
proc = subprocess.Popen(["python", "app.py"])
time.sleep(15)

print("Running batch evaluate...", flush=True)
subprocess.run(["python", "batch_evaluate_v500.py"])

print("Killing server...", flush=True)
proc.kill()
