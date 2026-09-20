import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    content = f.read()

patterns = [
    r"four\s+\w+\s+dataset",
    r"4\s+\w+\s+dataset",
    r"four\s+dataset",
    r"4\s+dataset",
    r"different\s+dataset",
    r"dataset\s+types",
    r"types\s+of\s+dataset",
    r"retinal\s+dataset",
    r"public\s+dataset",
    r"private\s+dataset"
]

print("--- PATTERN MATCHES ---")
for pattern in patterns:
    matches = re.finditer(pattern, content, re.IGNORECASE)
    for m in matches:
        start = max(0, m.start() - 100)
        end = min(len(content), m.end() + 200)
        snippet = content[start:end].replace('\n', ' ')
        print(f"[{pattern}]: {snippet}\n")
