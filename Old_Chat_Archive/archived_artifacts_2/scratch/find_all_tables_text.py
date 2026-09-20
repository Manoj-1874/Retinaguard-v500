import re

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Write surrounding text of tables to tables_text.txt
with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\tables_text.txt", "w", encoding="utf-8") as f_out:
    tables = re.finditer(r"Table\s+\d+", text, re.IGNORECASE)
    for idx, table in enumerate(tables):
        start_idx = max(0, table.start() - 200)
        end_idx = min(len(text), table.end() + 1500)
        f_out.write(f"\n--- MATCH {idx + 1}: {table.group()} SURROUNDING TEXT ---\n")
        f_out.write(text[start_idx:end_idx])
        f_out.write("\n" + "="*60 + "\n")

print("Done writing tables text.")
