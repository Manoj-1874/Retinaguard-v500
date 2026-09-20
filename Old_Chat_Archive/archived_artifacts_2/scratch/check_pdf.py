import re
import os

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"

print(f"Checking if file exists: {os.path.exists(pdf_path)}")
print(f"File size: {os.path.getsize(pdf_path)} bytes")

# Try to extract readable strings using regex on raw binary data
# This works well for PDF files because metadata is often in plain text XML or postscript streams.
try:
    with open(pdf_path, 'rb') as f:
        content = f.read(100 * 1024) # read first 100KB
        
        # Let's search for keywords like "Powroznik", "Retinitis", "Generative", "XGBoost", "VGG16"
        keywords = [b"Powro", b"Paszkowska", b"Nowomiejska", b"Aristidou", b"Rejdak", 
                    b"Retinitis", b"DCGAN", b"WGAN", b"VGG16", b"XGBoost", b"Advances in Science"]
        
        print("\n--- Key Term Matches in First 100KB ---")
        for kw in keywords:
            matches = list(re.finditer(re.escape(kw), content, re.IGNORECASE))
            print(f"Keyword '{kw.decode()}': {len(matches)} matches")
            
        # Try to find PDF /Title or XML metadata
        print("\n--- Extracting Title / Creator / Metadata blocks ---")
        # Find occurrences of standard XML tags like <dc:title>, <dc:creator>
        xml_title = re.findall(b'<dc:title[^>]*>(.*?)</dc:title>', content, re.IGNORECASE)
        for t in xml_title:
            print(f"XML Title: {t.decode(errors='ignore')}")
            
        xml_creator = re.findall(b'<dc:creator[^>]*>(.*?)</dc:creator>', content, re.IGNORECASE)
        for c in xml_creator:
            print(f"XML Creator/Author: {c.decode(errors='ignore')}")
            
        # Also scan the last 100KB of the file where PDF trailers/metadata often live
        f.seek(-min(100 * 1024, os.path.getsize(pdf_path)), 2)
        tail_content = f.read()
        
        print("\n--- Key Term Matches in Tail 100KB ---")
        for kw in keywords:
            matches = list(re.finditer(re.escape(kw), tail_content, re.IGNORECASE))
            print(f"Keyword '{kw.decode()}': {len(matches)} matches")
            
        xml_title_tail = re.findall(b'<dc:title[^>]*>(.*?)</dc:title>', tail_content, re.IGNORECASE)
        for t in xml_title_tail:
            print(f"XML Title (tail): {t.decode(errors='ignore')}")

except Exception as e:
    print(f"Error reading PDF: {e}")
