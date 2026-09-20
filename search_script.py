import json

path = r'E:\V500\Old_Chat_Archive\Part1_Logs\transcript_full_part1.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, ln in enumerate(lines):
    try:
        obj = json.loads(ln)
        if obj.get('type') == 'USER_INPUT':
            content = str(obj.get('content')).lower()
            if 'recover' in content or 'fine' in content or 'tune' in content or 'model' in content:
                print(f"--- STEP {i} ---")
                print(str(obj.get('content')))
    except Exception as e:
        pass
