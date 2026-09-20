import re

pdf_path = r"e:\V500\uploads\Deep-convolutional-genera.pdf"

keywords = [
    b"Powro", b"Paszkowska", b"Nowomiejska", b"Aristidou", b"Rejdak", 
    b"Retinitis", b"DCGAN", b"WGAN", b"VGG16", b"XGBoost", b"Advances in Science",
    b"Ophthalmology", b"Fundus", b"Sensors", b"Anaya", b"Sanchez", b"Rashid",
    b"Diabetic", b"Retinopathy", b"Glaucoma"
]

try:
    with open(pdf_path, 'rb') as f:
        content = f.read()
        
    print(f"Total file size read: {len(content)} bytes")
    
    print("\n--- Full File Scan ---")
    for kw in keywords:
        matches = list(re.finditer(re.escape(kw), content, re.IGNORECASE))
        print(f"Keyword '{kw.decode()}': {len(matches)} matches")
        
except Exception as e:
    print(f"Error: {e}")
