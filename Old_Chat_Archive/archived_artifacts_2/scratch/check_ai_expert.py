with open(r"e:\V500\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "DEEP_LEARNING_MODEL" in line:
        print(f"Line {idx + 1}: {line.strip()}")
