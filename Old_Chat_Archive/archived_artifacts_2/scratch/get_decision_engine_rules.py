with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("Decision Engine implementation (Lines 1500 to 1670):")
for i in range(1499, min(len(lines), 1670)):
    safe_line = lines[i].strip().encode('ascii', 'backslashreplace').decode('ascii')
    print(f"{i+1}: {safe_line}")
