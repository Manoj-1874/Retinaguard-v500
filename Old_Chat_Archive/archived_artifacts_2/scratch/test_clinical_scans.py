import os
import base64
import urllib.request
import json
import ssl

ssl_context = ssl._create_unverified_context()
uploads_dir = r"e:\V500\uploads"
api_url = "http://localhost:5001/api/analyze"

test_cases = [
    {
        "name": "Scenario 1: Sectoral Retinitis Pigmentosa",
        "file": "Sectoral_RP_Test.jpg",
        "patientId": "PT-SECTORAL-01",
        "cameraType": "Zeiss",
        "patient_history": {
            "age": 38,
            "ethnicity": "Caucasian",
            "symptoms": {
                "night_blindness": True,
                "tunnel_vision": True,
                "difficulty_dark_adaptation": True,
                "light_sensitivity": False,
                "color_vision_loss": False,
                "floaters": False
            },
            "family_history": True
        }
    },
    {
        "name": "Scenario 2: Fluorescein Angiography",
        "file": "Fluorescein_Angiography_Test.jpg",
        "patientId": "PT-ANGIO-02",
        "cameraType": "Generic",
        "patient_history": {
            "age": 52,
            "ethnicity": "South_Asian",
            "symptoms": {
                "night_blindness": False,
                "tunnel_vision": False,
                "difficulty_dark_adaptation": False,
                "light_sensitivity": True,
                "color_vision_loss": True,
                "floaters": True
            },
            "family_history": False
        }
    },
    {
        "name": "Scenario 3: Ultra-Widefield (Optomap) Stitched Retina",
        "file": "Ultra_Widefield_Optomap_Test.jpg",
        "patientId": "PT-WIDEFIELD-03",
        "cameraType": "Topcon",
        "patient_history": {
            "age": 29,
            "ethnicity": "Asian",
            "symptoms": {
                "night_blindness": True,
                "tunnel_vision": True,
                "difficulty_dark_adaptation": True,
                "light_sensitivity": True,
                "color_vision_loss": False,
                "floaters": False
            },
            "family_history": True
        }
    }
]

def run_tests():
    results = []
    print("=" * 80)
    print("RETINAGUARD V500 - CLINICAL SCAN VALIDATION RUNNER")
    print("=" * 80)
    
    for case in test_cases:
        filepath = os.path.join(uploads_dir, case["file"])
        if not os.path.exists(filepath):
            print(f"Error: Test image file not found at {filepath}")
            continue
            
        print(f"\n[RUNNING] {case['name']} ({case['file']})...")
        
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        data = {
            "image": "data:image/jpeg;base64," + encoded_string,
            "patientId": case["patientId"],
            "cameraType": case["cameraType"],
            "patient_history": case["patient_history"]
        }
        
        req = urllib.request.Request(
            api_url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, context=ssl_context) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                print(f"   [+] Analysis Status: Success")
                print(f"   [+] Verdict Code: {res_data.get('verdict_code')}")
                print(f"   [+] Verdict Text: {res_data.get('verdict')}")
                print(f"   [+] Confidence: {res_data.get('confidence')}")
                print(f"   [+] Composite Risk Score: {res_data.get('composite_score')}")
                print(f"   [+] Is Angiography: {res_data.get('is_angiography')}")
                
                if res_data.get('is_angiography'):
                    print(f"   [!] Angio Warning: {res_data.get('warning')}")
                    
                triad = res_data.get('triad_status', {})
                print(f"   [+] Triad Status:")
                print(f"       - Bone Spicules: {triad.get('bone_spicules')}")
                print(f"       - Vessel Attenuation: {triad.get('vessel_attenuation')}")
                print(f"       - Optic Disc Pallor: {triad.get('optic_disc_pallor')}")
                
                print(f"   [+] Expert Scanner Opinions:")
                for exp in res_data.get('expert_opinions', []):
                    print(f"       * {exp.get('name')}: status={exp.get('status')} | severity={exp.get('severity')} | conf={exp.get('confidence'):.1f}%")
                    
                results.append({
                    "case": case["name"],
                    "file": case["file"],
                    "status": "Success",
                    "data": res_data
                })
        except urllib.error.HTTPError as e:
            err_content = e.read().decode('utf-8')
            try:
                err_json = json.loads(err_content)
                print(f"   [X] Rejection/Error: {err_json.get('error')}")
                print(f"       Errors: {err_json.get('errors')}")
                print(f"       Recommendation: {err_json.get('recommendation')}")
                results.append({
                    "case": case["name"],
                    "file": case["file"],
                    "status": "Rejected/Error",
                    "error": err_json
                })
            except Exception:
                print(f"   [X] HTTP Error {e.code}: {e.reason}")
                print(f"       Content: {err_content[:200]}")
                results.append({
                    "case": case["name"],
                    "file": case["file"],
                    "status": "HTTP Error",
                    "error": err_content
                })
        except Exception as e:
            print(f"   [X] Failed: {e}")
            results.append({
                "case": case["name"],
                "file": case["file"],
                "status": "Failed",
                "error": str(e)
            })
            
    # Write out a markdown report artifact
    report_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\clinical_scans_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("# RetinaGuard V500 - Clinical Scan Validation Report\n\n")
        f_rep.write("This report documents the validation of three critical diagnostic scenarios using curated clinical test scans.\n\n")
        
        for res in results:
            f_rep.write(f"## {res['case']} ({res['file']})\n")
            f_rep.write(f"**Execution Status:** {res['status']}\n\n")
            
            if res['status'] == "Success":
                d = res['data']
                f_rep.write(f"- **Final Verdict:** `{d.get('verdict_code')}` ({d.get('verdict')})\n")
                f_rep.write(f"- **Confidence:** {d.get('confidence')}\n")
                f_rep.write(f"- **Composite Risk Score:** {d.get('composite_score', 0)*100:.1f}%\n")
                f_rep.write(f"- **Is Angiography:** {d.get('is_angiography')}\n")
                if d.get('is_angiography'):
                    f_rep.write(f"  - *Warning:* `{d.get('warning')}`\n")
                    
                f_rep.write("\n### Classical RP Triad Status\n")
                t = d.get('triad_status', {})
                f_rep.write(f"- **Bone Spicules (Triad #1):** {t.get('bone_spicules')}\n")
                f_rep.write(f"- **Vessel Attenuation (Triad #2):** {t.get('vessel_attenuation')}\n")
                f_rep.write(f"- **Optic Disc Pallor (Triad #3):** {t.get('optic_disc_pallor')}\n\n")
                
                f_rep.write("### Multi-Agent Expert Scanner Opinions\n")
                f_rep.write("| Scanner Module | Diagnostic Status | Severity | Confidence | Details |\n")
                f_rep.write("| --- | --- | --- | --- | --- |\n")
                for exp in d.get('expert_opinions', []):
                    f_rep.write(f"| {exp.get('name')} | {exp.get('status')} | {exp.get('severity')} | {exp.get('confidence'):.1f}% | {exp.get('detail', '')} |\n")
                f_rep.write("\n")
            else:
                f_rep.write("### Rejection / Error Details\n")
                f_rep.write("```json\n")
                f_rep.write(json.dumps(res.get('error'), indent=2))
                f_rep.write("\n```\n\n")
                
    print(f"\n[COMPLETE] Written markdown report artifact to: {report_path}")

if __name__ == "__main__":
    run_tests()
