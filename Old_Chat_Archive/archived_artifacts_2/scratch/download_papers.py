import urllib.request
import ssl
import os

ssl_context = ssl._create_unverified_context()

uploads_dir = r"e:\V500\uploads"
os.makedirs(uploads_dir, exist_ok=True)

# Use the PMC direct PDF URL
pmc_pdf_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11723073/pdf/"
dest_path = os.path.join(uploads_dir, "WGAN_GP_Retinal_Augmentation_Sensors_2024.pdf")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"Downloading from PMC: {pmc_pdf_url}...")
try:
    req = urllib.request.Request(pmc_pdf_url, headers=headers)
    with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Successfully saved to: {dest_path} ({len(data)} bytes)")
except Exception as e:
    print(f"Failed to download from PMC: {e}")
