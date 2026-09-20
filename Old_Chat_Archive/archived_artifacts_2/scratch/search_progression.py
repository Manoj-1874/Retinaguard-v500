with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if "progression-compare" in line:
        print(f"Line {i}: {line.strip()}")
        # Let's print 50 lines starting here
        for idx in range(i - 5, min(len(lines), i + 65)):
            safe_line = lines[idx].strip().encode('ascii', 'backslashreplace').decode('ascii')
            print(f"  {idx+1}: {safe_line}")
        break
