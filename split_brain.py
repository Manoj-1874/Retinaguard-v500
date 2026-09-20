import os
import json
import shutil
import sys

brain_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7"
logs_dir = os.path.join(brain_dir, ".system_generated", "logs")

if not os.path.exists(logs_dir):
    print(f"Error: Could not find logs at {logs_dir}")
    sys.exit(1)

# 1. Create a safe backup first
backup_dir = os.path.join(brain_dir, ".system_generated", "logs_backup")
if not os.path.exists(backup_dir):
    shutil.copytree(logs_dir, backup_dir)
    print(f"Backup successfully created at: {backup_dir}")

def split_transcript(filename):
    file_path = os.path.join(logs_dir, filename)
    if not os.path.exists(file_path):
        return
        
    part1_path = os.path.join(logs_dir, f"{filename.split('.')[0]}_part1.jsonl")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if len(lines) < 200:
        print(f"{filename} is already small ({len(lines)} lines). Skipping split.")
        return
        
    # Keep first 20 lines (initial context/system prompts)
    # Keep last 50 lines (immediate active memory)
    head = lines[:20]
    middle = lines[20:-50]
    tail = lines[-50:]
    
    # 2. Save the massive middle section into the part1 archive
    with open(part1_path, 'w', encoding='utf-8') as f:
        f.writelines(middle)
        
    # 3. Create a reference pointer for the AI
    ref_step = {
        "step_index": 999999, # Dummy index
        "source": "SYSTEM",
        "type": "USER_INPUT",
        "content": f"[SYSTEM: Older conversation history was safely partitioned and archived to {part1_path} to optimize UI rendering speed.]"
    }
    
    # 4. Overwrite the active file with only head + reference + tail
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(head)
        f.write(json.dumps(ref_step) + "\n")
        f.writelines(tail)
        
    print(f"Successfully partitioned {filename}. Moved {len(middle)} lines to archive.")

try:
    split_transcript("transcript.jsonl")
    split_transcript("transcript_full.jsonl")
    print("\nSUCCESS: The brain has been safely partitioned without data loss.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
