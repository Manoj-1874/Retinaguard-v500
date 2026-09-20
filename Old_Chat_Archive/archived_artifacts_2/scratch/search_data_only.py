import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Let's find occurrences of "data" that do not have "set" or "sets" or "base" right after them
data_mentions = re.finditer(r"\bdata\b", content, re.IGNORECASE)
with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\data_mentions.txt", "w", encoding="utf-8") as f_out:
    for idx, match in enumerate(data_mentions):
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 150)
        snippet = content[start:end].replace('\n', ' ')
        f_out.write(f"Data Mention {idx + 1}: ... {snippet} ...\n\n")

print("Done. Wrote data mentions.")
