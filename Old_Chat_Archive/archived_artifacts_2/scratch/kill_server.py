import os
import signal

pids = [3468]
for pid in pids:
    print(f"Terminating process {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  Sent SIGTERM to {pid}")
    except Exception as e:
        print(f"  SIGTERM to {pid} failed: {e}")
        
    try:
        os.kill(pid, 9)
        print(f"  Sent SIGKILL to {pid}")
    except Exception as e:
        print(f"  SIGKILL to {pid} failed: {e}")
