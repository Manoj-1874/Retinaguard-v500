import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'DEEP_LEARNING_MODEL' in line and 'predict' in line or 'training=False' in line:
        start = max(0, i - 15)
        end = min(len(lines), i + 20)
        print(''.join(lines[start:end]))
        break
