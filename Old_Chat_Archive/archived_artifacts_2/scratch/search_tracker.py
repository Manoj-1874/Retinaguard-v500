with open(r"e:\V500\progression_tracker.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

search_terms = ["baseline_img", "current_img", "months_between", "track_progression"]
for i, line in enumerate(lines, 1):
    for term in search_terms:
        if term.lower() in line.lower():
            safe_line = line.strip().encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Line {i} ({term}): {safe_line}")
