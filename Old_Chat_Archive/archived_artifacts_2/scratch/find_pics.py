import urllib.request
import re
import ssl

ssl_context = ssl._create_unverified_context()
url = "https://webeye.ophth.uiowa.edu/eyeforum/atlas/pages/sectoral-retinitis-pigmentosa.htm"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ssl_context) as response:
        html = response.read().decode('utf-8')
    # Find all image sources
    img_srcs = re.findall(r'src=["\']([^"\']+)["\']', html)
    print("Found images on Sectoral RP page:")
    for src in img_srcs:
        print(src)
except Exception as e:
    print(f"Error fetching page: {e}")
