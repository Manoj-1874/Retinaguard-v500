import os
import subprocess
import time

print("Starting app.py...")
proc = subprocess.Popen(["python", "app.py"])

print("Waiting for server...")
time.sleep(15)

print("Running batch evaluate...")
subprocess.run(["python", "batch_evaluate_v500.py"])

print("Killing server...")
proc.kill()
