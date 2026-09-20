import sys

with open(r"e:\V500\camera_calibrator.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

search_terms = ["optos", "widefield", "vignette", "vignetting"]
for i, line in enumerate(lines, 1):
    for term in search_terms:
        if term.lower() in line.lower():
            # Print safely without emojis crashing standard output on Windows CP1252
            safe_line = line.strip().encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Line {i} ({term}): {safe_line}")
