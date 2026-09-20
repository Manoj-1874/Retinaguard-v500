import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Let's find sections or paragraphs that mention datasets.
# We can search for the tables
tables = re.finditer(r"Table\s+\d+", content, re.IGNORECASE)
print("--- TABLES ---")
for t in tables:
    start = max(0, t.start() - 100)
    end = min(len(content), t.end() + 300)
    snippet = content[start:end].replace('\n', ' ')
    print(f"[{t.group()}]: {snippet}\n")
