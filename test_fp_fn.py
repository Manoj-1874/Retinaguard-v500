import requests
import json
import os
import base64

URL = "http://localhost:5001/api/analyze"

test_images = [
    "Dataset/Original Dataset/Retinitis Pigmentosa/Retinitis Pigmentosa55.jpg",
    "Dataset/Original Dataset/Retinitis Pigmentosa/Retinitis Pigmentosa12.jpg",
    "Dataset/Original Dataset/Retinitis Pigmentosa/Retinitis Pigmentosa30.jpg",
    "Dataset/Original Dataset/Retinitis Pigmentosa/Retinitis Pigmentosa135.jpg",
    "Dataset/Original Dataset/Healthy/Healthy183.jpg",
    "Dataset/Original Dataset/Healthy/Healthy162.jpg",
    "Dataset/Original Dataset/Healthy/Healthy189.jpg",
    "Dataset/Original Dataset/Healthy/Healthy148.jpg",
]

def image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    ext = os.path.splitext(filepath)[1].lower().replace('.', '')
    mime = 'jpeg' if ext == 'jpg' else ext
    return f"data:image/{mime};base64,{encoded_string}"

for img in test_images:
    abs_path = os.path.abspath(img)
    try:
        b64 = image_to_base64(abs_path)
        payload = {
            "image": b64,
            "patientId": img.split('/')[-1],
            "bypassQualityCheck": True,
            "patient_history": {}
        }
        resp = requests.post(URL, json=payload, timeout=30)
        if resp.status_code == 200:
            res = resp.json()
            print(f"--- {img.split('/')[-1]} ---")
            print(json.dumps(res, indent=2))
        else:
            print(f"{img.split('/')[-1]}: Server Error {resp.status_code}")
    except Exception as e:
        print(f"Error testing {img}: {e}")
