# ==============================================================================
# COLAB SCRIPT: V500 LIVE SENSITIVITY TEST
# ==============================================================================
# Copy and paste this script into a Google Colab cell.
# It will connect to your local app.py via the secure tunnel.
# ==============================================================================

import requests
import base64
import os
import glob
from google.colab import drive

# 1. MOUNT GOOGLE DRIVE
print("🔄 Mounting Google Drive...")
drive.mount('/content/drive')

# 2. CONFIGURATION
# This is the secure tunnel pointing directly to your local Windows app.py
API_URL = "https://7c4765ace1dbc2.lhr.life/api/analyze"
HEADERS = {} 

def encode_image_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_remote_sensitivity():
    print("\n==================================================")
    print("🛡️ VERIFYING V500 SENSITIVITY VIA SECURE TUNNEL")
    print("==================================================")
    
    print("🔍 Performing deep search of Google Drive for RP images...")
    
    # Search exact paths with the correct space in the folder name
    rp_images = []
    base_dirs = [
        '/content/drive/MyDrive/Dataset/Train/Retinitis Pigmentosa',
        '/content/drive/MyDrive/Dataset/Test/Retinitis Pigmentosa',
        '/content/drive/MyDrive/Dataset/Valid/Retinitis Pigmentosa'
    ]
    
    for d in base_dirs:
        rp_images.extend(glob.glob(f'{d}/*.jpg'))
        rp_images.extend(glob.glob(f'{d}/*.jpeg'))
        rp_images.extend(glob.glob(f'{d}/*.png'))
            
    if not rp_images:
        print("❌ Error: Deep search failed. Could not find any RP images.")
        return
        
    print(f"✅ SUCCESS! Found {len(rp_images)} true RP images.")
    print(f"Testing a subset of up to 100 images...\n")
    test_limit = min(100, len(rp_images))
    
    true_positives = 0
    false_negatives = 0
    
    for i, img_path in enumerate(rp_images[:test_limit]):
        img_name = os.path.basename(img_path)
        
        try:
            b64_img = encode_image_to_base64(img_path)
            
            payload = {
                "image": b64_img,
                "patient_history": {
                    "age": 45,
                    "ethnicity": "unknown"
                }
            }
            
            print(f"[{i+1}/{test_limit}] Routing {img_name} to Windows AI...")
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                diagnosis = data.get('diagnosis', '')
                
                if any(keyword in diagnosis.upper() for keyword in ["HEALTHY", "NEGATIVE"]):
                    false_negatives += 1
                    print(f"   ❌ [FALSE NEGATIVE] Incorrectly rejected! ({diagnosis})")
                else:
                    true_positives += 1
                    print(f"   ✅ [TRUE POSITIVE] RP Detected. ({diagnosis})")
            else:
                print(f"   ⚠️ Tunnel Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Connection Failed: {e}")

    total_tested = true_positives + false_negatives
    sensitivity = (true_positives / total_tested) * 100 if total_tested > 0 else 0
    
    print("\n==================================================")
    print("🏆 FINAL V500 SENSITIVITY RESULTS")
    print("==================================================")
    print(f"Total RP Images Tested: {total_tested}")
    print(f"True Positives (RP Detected): {true_positives}")
    print(f"False Negatives (Missed RP): {false_negatives}")
    print(f"--------------------------------------------------")
    print(f"V500 SENSITIVITY: {sensitivity:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    test_remote_sensitivity()
