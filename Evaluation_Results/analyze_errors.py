import json
import re

with open('V500_Evaluation_Results.json', 'r') as f:
    results = json.load(f)

# Find all FNs
fns = [r for r in results['results'] if r['is_error'] and r['true_label'] == 'RP_CONFIRMED']
print("FNs:")
for fn in fns:
    print(f"{fn['image']} - AI: {fn['ai_prob']}%")

# Find all FPs
fps = [r for r in results['results'] if r['is_error'] and r['true_label'] == 'HEALTHY']
print("\nFPs:")
for fp in fps:
    print(f"{fp['image']} - AI: {fp['ai_prob']}%")

