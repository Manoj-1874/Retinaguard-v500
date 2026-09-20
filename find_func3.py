import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'DEEP_LEARNING_MODEL = keras.models.load_model' in line:
        start = max(0, i - 10)
        end = min(len(lines), i + 5)
        for j in range(start, end):
            print(f"{j+1}: {lines[j]}", end='')
        break
