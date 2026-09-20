import json

path = r'E:\V500\Old_Chat_Archive\Part1_Logs\transcript_full_part1.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

user_inputs = []
for i, ln in enumerate(lines):
    try:
        obj = json.loads(ln)
        if obj.get('type') == 'USER_INPUT':
            content = str(obj.get('content'))
            if not content.startswith('<USER_REQUEST>'):
                continue
            # just get the content inside <USER_REQUEST> tags
            req = content.split('<USER_REQUEST>')[1].split('</USER_REQUEST>')[0].strip()
            user_inputs.append(f"[{i}] {req}")
    except Exception as e:
        pass

with open('user_inputs_dump.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(user_inputs))
