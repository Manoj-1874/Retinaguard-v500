with open(r"e:\V500\public\index.html", "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()

# Let's find script tags
import re
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
print(f"Found {len(scripts)} inline script tags.")

for idx, script in enumerate(scripts):
    print(f"--- Script {idx+1} (Length: {len(script)}) ---")
    # Check for while, for, function calls
    lines = script.splitlines()
    for l_no, line in enumerate(lines, 1):
        if any(term in line for term in ["while", "for", "recursive", "interval", "timeout"]):
            print(f"  Line {l_no}: {line.strip()}")
