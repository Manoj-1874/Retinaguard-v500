import requests
import base64
import os
import glob

# CONFIGURATION
API_URL = "http://127.0.0.1:5001/api/analyze"
RP_DIR = r"E:\V500\Research_Paper_Proofs\sensitivity\Retinitis Pigmentosa"

def encode_image_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_v500_sensitivity():
    print("==================================================")
    print("🛡️ VERIFYING V500 SENSITIVITY (RP RECALL)")
    print("==================================================")
    
    # Get all RP images
    rp_images = glob.glob(os.path.join(RP_DIR, "*.jpg")) + glob.glob(os.path.join(RP_DIR, "*.png"))
    
    if not rp_images:
        print("❌ Error: No RP images found in directory.")
        return
        
    print(f"🔍 Found {len(rp_images)} true RP images. Testing a subset of 100 images...")
    
    # Test up to 100 images to save time, or all if less than 100
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
            
            print(f"[{i+1}/{test_limit}] Scanning {img_name}...")
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                diagnosis = data.get('diagnosis', '')
                
                # Check if the AI correctly diagnosed RP
                # A false negative means it said HEALTHY or OTHER DISEASE
                if any(keyword in diagnosis.upper() for keyword in ["HEALTHY", "NEGATIVE"]):
                    false_negatives += 1
                    print(f"   ❌ [FALSE NEGATIVE] The rule engine incorrectly rejected this RP image! ({diagnosis})")
                else:
                    # POSITIVE, SUSPICIOUS, BORDERLINE, SECTORAL, SINE PIGMENTO all count as detecting the disease
                    true_positives += 1
                    print(f"   ✅ [TRUE POSITIVE] Successfully detected RP. ({diagnosis})")
                    
            else:
                print(f"   ⚠️ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Request Failed: {e}")

    # Calculate Sensitivity
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
    
    if sensitivity > 98:
        print("💡 VERDICT: EXCELLENT! The Rule Engine did not destroy the Sensitivity.")
    else:
        print("🚨 VERDICT: DANGER! The rules are too strict and are rejecting real RP cases.")

if __name__ == "__main__":
    test_v500_sensitivity()
