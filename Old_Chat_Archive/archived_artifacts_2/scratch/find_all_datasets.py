import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Let's search for "dataset", "data set", "database", "databases"
matches = re.finditer(r"(?:dataset|data set|database|databases|Mendeley|DRIVE|STARE|MESSIDOR|KAGGLE|RFMiD|EyePACS|IDRiD|HRF|ARIA|CHASE|OIA|PALM)", content, re.IGNORECASE)

output_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\dataset_matches.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for idx, match in enumerate(matches):
        start = max(0, match.start() - 150)
        end = min(len(content), match.end() + 250)
        snippet = content[start:end].replace('\n', ' ')
        f.write(f"Match {idx + 1} [{match.group()}]: ... {snippet} ...\n\n")

print("Done searching. Output written to dataset_matches.txt")
