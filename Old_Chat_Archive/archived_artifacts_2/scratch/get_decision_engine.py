import sys
sys.path.append(r"e:\V500")

with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    code = f.read()

# Let's find lines with "Decision Engine" or "Simplified 6-Rule System"
lines = code.splitlines()
start_line = -1
for idx, line in enumerate(lines):
    if "Simplified 6-Rule System" in line or "DECISION ENGINE" in line:
        start_line = idx
        break

if start_line != -1:
    print(f"Found Decision Engine starting at line {start_line + 1}:")
    for i in range(max(0, start_line - 10), min(len(lines), start_line + 120)):
        safe_line = lines[i].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"{i+1}: {safe_line}")
else:
    print("Decision Engine section not found by name search.")
