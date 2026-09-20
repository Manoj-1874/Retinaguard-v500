import PyPDF2

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"
reader = PyPDF2.PdfReader(pdf_path)

print("--- PAGE 1 FULL TEXT ---")
text = reader.pages[0].extract_text()
print(text)
