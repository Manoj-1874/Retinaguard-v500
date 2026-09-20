import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = r'E:\V500\user_inputs_dump.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inputs = content.split('\n[')
for inp in inputs:
    if 'recover' in inp.lower() and ('fine' in inp.lower() or 'train' in inp.lower() or 'colab' in inp.lower()):
        print('--- MATCH ---')
        print(inp[:300])
