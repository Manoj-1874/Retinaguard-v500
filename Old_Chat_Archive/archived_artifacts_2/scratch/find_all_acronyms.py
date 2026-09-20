import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Find words that consist of 3-10 uppercase letters
acronyms = set(re.findall(r"\b[A-Z]{3,10}\b", text))
print("All uppercase acronyms found in PDF:")
print(sorted(list(acronyms)))
