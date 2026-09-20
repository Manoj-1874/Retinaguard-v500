import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
path = r'E:\V500\user_inputs_dump.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inputs = content.split('\n[')
for i, inp in enumerate(inputs):
    if 'why finetuneed cant we do it on recovered' in inp:
        print('--- MESSAGE ---')
        print(inp)
