import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "four" or "4" and print surrounding text if it has anything to do with datasets or studies
matches = re.finditer(r"\b(?:four|4)\b", text, re.IGNORECASE)
with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\four_mentions.txt", "w", encoding="utf-8") as f_out:
    for idx, match in enumerate(matches):
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 200)
        snippet = text[start:end].replace('\n', ' ')
        if any(w in snippet.lower() for w in ["data", "set", "patient", "imag", "stud", "refer", "method"]):
            f_out.write(f"Mention {idx + 1}: ... {snippet} ...\n\n")

print("Done. Check four_mentions.txt")
