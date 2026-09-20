with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

import re
ref_match = re.search(r"REFERENCES", text)
if ref_match:
    with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\more_references.txt", "w", encoding="utf-8") as f_out:
        f_out.write(text[ref_match.start() + 4000:ref_match.start() + 15000])
    print("Done writing references.")
else:
    print("REFERENCES not found")
