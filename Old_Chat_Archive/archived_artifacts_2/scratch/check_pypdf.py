import sys

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"

try:
    import pypdf
    print("pypdf is installed!")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    
    # Extract metadata
    meta = reader.metadata
    print("\n--- Metadata ---")
    if meta:
        for k, v in meta.items():
            print(f"{k}: {v}")
    else:
        print("No metadata found.")
        
    # Extract text from first 2 pages
    print("\n--- Text from page 1 & 2 ---")
    for i in range(min(2, len(reader.pages))):
        print(f"\n--- PAGE {i+1} ---")
        text = reader.pages[i].extract_text()
        print(text[:1000]) # print first 1000 chars
        
except ImportError:
    print("pypdf not installed. Trying PyPDF2...")
    try:
        import PyPDF2
        print("PyPDF2 is installed!")
        reader = PyPDF2.PdfReader(pdf_path)
        print(f"Number of pages: {len(reader.pages)}")
        meta = reader.metadata
        print("\n--- Metadata ---")
        if meta:
            for k, v in meta.items():
                print(f"{k}: {v}")
        for i in range(min(2, len(reader.pages))):
            print(f"\n--- PAGE {i+1} ---")
            text = reader.pages[i].extract_text()
            print(text[:1000])
    except ImportError:
        print("Neither pypdf nor PyPDF2 is installed.")
        # Let's inspect objects in the PDF using standard file parser
        with open(pdf_path, 'rb') as f:
            head = f.read(500)
            print(f"\nPDF Header:\n{head.decode(errors='ignore')}")
