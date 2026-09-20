import urllib.request
import ssl
import os
import re

ssl_context = ssl._create_unverified_context()
uploads_dir = r"e:\V500\uploads"
os.makedirs(uploads_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

wiki_url = "https://commons.wikimedia.org/wiki/File:Fundus_of_patient_with_retinitis_pigmentosa,_end_stage.jpg"
output_filename = "Sectoral_RP_Test.jpg"
dest_path = os.path.join(uploads_dir, output_filename)

print("Fetching Wikimedia page for End-Stage/Sectoral Retinitis Pigmentosa fundus image...")
try:
    req = urllib.request.Request(wiki_url, headers=headers)
    with urllib.request.urlopen(req, context=ssl_context) as response:
        html = response.read().decode('utf-8')
    
    matches = re.findall(r'class="fullMedia".*?href="([^"]+)"', html, re.DOTALL)
    if matches:
        direct_url = matches[0]
        print(f"Found direct URL: {direct_url}")
        print(f"Downloading image to {output_filename}...")
        img_req = urllib.request.Request(direct_url, headers=headers)
        with urllib.request.urlopen(img_req, context=ssl_context) as img_resp, open(dest_path, 'wb') as f_out:
            f_out.write(img_resp.read())
        print(f"Successfully saved {output_filename}")
    else:
        print("Could not find direct URL in HTML.")
except Exception as e:
    print(f"Failed to download image: {e}")
