import PyPDF2
import re

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"
output_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\extracted_limitations.txt"

reader = PyPDF2.PdfReader(pdf_path)

with open(output_path, "w", encoding="utf-8") as out:
    out.write("=== EXTRACTED LIMITATIONS & DISCUSSION ===\n\n")
    
    # We will search the last 5 pages specifically (Pages 15-20)
    for i in range(len(reader.pages) - 5, len(reader.pages)):
        out.write(f"\n================ PAGE {i+1} ================\n")
        text = reader.pages[i].extract_text()
        out.write(text)
        out.write("\n\n")

print(f"Extracted content written to {output_path}")
