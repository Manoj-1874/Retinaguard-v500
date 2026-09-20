with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i in range(1669, min(len(lines), 1750)):
    safe_line = lines[i].strip().encode('ascii', 'backslashreplace').decode('ascii')
    print(f"{i+1}: {safe_line}")
