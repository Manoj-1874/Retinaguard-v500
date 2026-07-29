import requests
import json
import base64

def run_early_stage_test():
    url = "http://localhost:5001/api/analyze"
    
    # We will use the healthy image we just generated
    image_path = "uploads/healthy_retina_edge_case.png"
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to read image: {e}")
        return

    # To trigger the new Rule 6b, we need a high clinical risk score.
    # We will simulate a patient who has severe night blindness and a family history of RP.
    payload = {
        "image": f"data:image/png;base64,{encoded_string}",
        "patient_id": "PT-EARLY-STAGE",
        "patient_history": {
            "age": 22,
            "ethnicity": "caucasian",
            "symptoms": {
                "night_blindness": True,
                "tunnel_vision": True,
                "glare_sensitivity": True
            },
            "family_history": True,
            "visual_field_data": None
        }
    }

    print("Sending API Request to RetinaGuard V500...")
    print("Patient Profile: 22yo, Severe Night Blindness, Positive Family History of RP.")
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        print("\n" + "="*50)
        print("API RESPONSE")
        print("="*50)
        print(f"Verdict: {result.get('verdict')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Score: {result.get('risk_score', result.get('score'))}")
        print(f"Patient Data Seen By Server: {result.get('patient_analysis')}")
        print(f"\nXAI Explanation:\n{result.get('xai_explanation')}")
        print("="*50)
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    run_early_stage_test()
