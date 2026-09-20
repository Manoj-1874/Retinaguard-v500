def extract_results():
    import json
    with open('V500_Evaluation_Results.json', 'r') as f:
        data = json.load(f)
    print("Keys in JSON:", data.keys())

extract_results()
