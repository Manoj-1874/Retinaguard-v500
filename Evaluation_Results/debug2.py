import json
with open('V500_Evaluation_Results.json', 'r') as f:
    data = json.load(f)

for result in data['per_image_results']:
    if result['true_label'] == 'RP_CONFIRMED' and result['predicted_label'] == 'HEALTHY':
        print(f"{result['image']} - AI: {result['ai_prob']}%")
