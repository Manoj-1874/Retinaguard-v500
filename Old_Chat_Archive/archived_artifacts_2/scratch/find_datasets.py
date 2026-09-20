import PyPDF2

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"
reader = PyPDF2.PdfReader(pdf_path)

with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_pdf_text.txt", "w", encoding="utf-8") as f:
    for idx, page in enumerate(reader.pages):
        f.write(f"\n--- PAGE {idx + 1} ---\n")
        f.write(page.extract_text() or "")
print("Done writing PDF text to file.")
