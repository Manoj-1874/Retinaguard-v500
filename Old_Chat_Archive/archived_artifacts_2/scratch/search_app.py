import re

with open(r"e:\V500\app.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

search_terms = ["/api/analyze", "expert_opinions", "quadrant", "sectoral", "vignette", "angio"]
for i, line in enumerate(lines, 1):
    for term in search_terms:
        if term.lower() in line.lower():
            print(f"Line {i} ({term}): {line.strip()}")
