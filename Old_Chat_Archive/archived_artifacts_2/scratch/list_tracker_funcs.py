with open(r"e:\V500\progression_tracker.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if line.startswith("def ") or "class " in line:
        print(f"Line {i}: {line.strip()}")
