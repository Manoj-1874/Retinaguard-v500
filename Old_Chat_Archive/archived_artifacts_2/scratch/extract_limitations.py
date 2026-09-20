import PyPDF2
import re

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"
reader = PyPDF2.PdfReader(pdf_path)

print(f"Reading PDF with {len(reader.pages)} pages...")

# Scan all pages for keywords
matches = []
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    # Search for headings or paragraphs containing keywords
    keywords = ["limitation", "future work", "conclusions", "discussion", "future direction"]
    for kw in keywords:
        for match in re.finditer(r'(?i)\b' + re.escape(kw) + r'\b', text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 1500) # grab 1500 chars after the match
            matches.append((idx + 1, kw, text[start:end]))

# Print page-by-page findings for discussion and conclusion
print("\n--- EXTRACTING DISCUSSIONS / LIMITATIONS / CONCLUSIONS ---")
# Let's extract the full text of the last few pages (typically pages 15-20) where discussion/conclusion/limitations reside.
for i in range(len(reader.pages) - 4, len(reader.pages)):
    print(f"\n================ PAGE {i+1} ================")
    text = reader.pages[i].extract_text()
    
    # Print lines that look like headers or contain discussion/limitation/future
    lines = text.split('\n')
    for line in lines[:10]: # Print first 10 lines of each page to find headers
        print(f"Header candidates: {line}")
        
    # Print text that matches "limitation" or "future" or "conclusions"
    print("\n--- Relevant Text Snippets ---")
    matches_on_page = []
    keywords_on_page = ["limitation", "future", "conclusion", "challenge"]
    for kw in keywords_on_page:
        for m in re.finditer(r'(?i)\b' + re.escape(kw) + r'\b', text):
            # Print surrounding paragraph (approx 500 chars before and 1000 chars after)
            start_p = max(0, m.start() - 300)
            end_p = min(len(text), m.end() + 800)
            snippet = text[start_p:end_p]
            print(f"[{kw.upper()} MATCH]:\n{snippet}\n{'-'*50}")
            break # only print one per page per keyword to avoid clutter
