import os
import json
import sys

brain_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain"
current_id = "8216f9b3-0dde-48a9-8bb3-6627b661dec0"

# Find all valid transcripts
valid_transcripts = []
try:
    for d in os.listdir(brain_dir):
        if d == current_id: continue
        dir_path = os.path.join(brain_dir, d)
        if not os.path.isdir(dir_path): continue
        
        transcript_path = os.path.join(dir_path, ".system_generated", "logs", "transcript.jsonl")
        if os.path.exists(transcript_path):
            valid_transcripts.append(transcript_path)
except Exception as e:
    print(f"Error accessing brain dir: {e}")
    sys.exit(1)

if not valid_transcripts:
    print("No previous chats found.")
    sys.exit(1)

# Get the most recently modified transcript file
latest_transcript = max(valid_transcripts, key=os.path.getmtime)
print(f"Reading from: {latest_transcript}")

chat_history = []
try:
    with open(latest_transcript, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                step_type = data.get("type")
                if step_type in ["USER_INPUT", "PLANNER_RESPONSE"]:
                    sender = "USER" if step_type == "USER_INPUT" else "AI"
                    content = data.get("content", "")
                    if content:
                        chat_history.append(f"--- {sender} ---\n{content}\n")
            except Exception as e:
                pass
except Exception as e:
    print(f"Error reading transcript: {e}")
    sys.exit(1)

output_path = r"e:\V500\recovered_chats.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n\n".join(chat_history))
print(f"Recovered chat history written to {output_path}")