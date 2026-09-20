import urllib.request
import re
import ssl
import os

ssl_context = ssl._create_unverified_context()
uploads_dir = r"e:\V500\uploads"
os.makedirs(uploads_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

files = [
    {
        "wiki_url": "https://commons.wikimedia.org/wiki/File:Fluorescein_angiography.jpg",
        "output_filename": "Fluorescein_Angiography_Test.jpg",
        "description": "Fluorescein Angiography (Black and White)"
    },
    {
        "wiki_url": "https://commons.wikimedia.org/wiki/File:Lefteyeoptomap.jpg",
        "output_filename": "Ultra_Widefield_Optomap_Test.jpg",
        "description": "Ultra-Widefield (Optomap) Retina"
    }
]

for item in files:
    print(f"Fetching Wikimedia page for {item['description']}...")
    try:
        req = urllib.request.Request(item["wiki_url"], headers=headers)
        with urllib.request.urlopen(req, context=ssl_context) as response:
            html = response.read().decode('utf-8')
        
        # Search for the direct image link in the fullMedia div
        # e.g., <div class="fullMedia"><p><a href="https://upload.wikimedia.org/wikipedia/commons/.../..."
        matches = re.findall(r'class="fullMedia".*?href="([^"]+)"', html, re.DOTALL)
        if matches:
            direct_url = matches[0]
            print(f"Found direct URL: {direct_url}")
            
            # Download the image
            print(f"Downloading image to {item['output_filename']}...")
            img_req = urllib.request.Request(direct_url, headers=headers)
            with urllib.request.urlopen(img_req, context=ssl_context) as img_resp, open(os.path.join(uploads_dir, item['output_filename']), 'wb') as f_out:
                f_out.write(img_resp.read())
            print(f"Successfully saved {item['output_filename']}\n")
        else:
            print(f"Could not find direct URL for {item['output_filename']}\n")
    except Exception as e:
        print(f"Error processing {item['description']}: {e}\n")
