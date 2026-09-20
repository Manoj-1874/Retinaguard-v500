with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("Expert weights configuration (Lines 110 to 140):")
for i in range(109, min(len(lines), 140)):
    safe_line = lines[i].strip().encode('ascii', 'backslashreplace').decode('ascii')
    print(f"{i+1}: {safe_line}")
