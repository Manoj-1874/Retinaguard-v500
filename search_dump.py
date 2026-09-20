path = r'E:\V500\user_inputs_dump.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inputs = content.split('\n[')
for inp in inputs:
    if 'recover' in inp.lower() or 'fine' in inp.lower():
        print('--- MATCH ---')
        print(inp[:500])
