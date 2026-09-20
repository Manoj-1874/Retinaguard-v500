import json
path = r'E:\V500\Old_Chat_Archive\Part1_Logs\transcript_full_part1.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

user_inputs = []
responses = []

for ln in lines:
    try:
        obj = json.loads(ln)
        if obj.get('type') == 'USER_INPUT':
            user_inputs.append(obj.get('content'))
        elif obj.get('type') == 'PLANNER_RESPONSE' and obj.get('content'):
            responses.append(obj.get('content'))
    except Exception as e:
        pass

print('--- LAST 2 USER INPUTS ---')
for u in user_inputs[-2:]:
    print(str(u)[:2000])

print('--- LAST RESPONSE ---')
if responses:
    print(str(responses[-1])[:2000])
