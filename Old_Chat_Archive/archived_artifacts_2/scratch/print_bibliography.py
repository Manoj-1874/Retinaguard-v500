with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "REFERENCES"
import re
ref_match = re.search(r"REFERENCES", text)
if ref_match:
    print(text[ref_match.start():ref_match.start() + 5000])
else:
    print("REFERENCES not found")
