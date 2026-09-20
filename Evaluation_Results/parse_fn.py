import re

log_content = open('../retinaguard_analysis.log', 'r').read()

images = log_content.split('----------------------------------------------------------------------')
print(f"Total image blocks: {len(images)}")

for img_block in images:
    if 'Image:' not in img_block: continue
    
    img_name_match = re.search(r'Image:\s*(.*?)\s+', img_block)
    if not img_name_match: continue
    img_name = img_name_match.group(1).split('\\')[-1].split('/')[-1]
    
    # We care about FNs (RP... -> HEALTHY) and FPs (Healthy... -> NOT HEALTHY)
    if img_name.startswith('Retinitis') and '[V] VERDICT: HEALTHY' in img_block:
        # It's an FN
        ai_match = re.search(r'AI Pattern Recognition.*?confidence\': ([\d\.]+)', img_block)
        votes_match = re.search(r'Clinical Votes \(MODERATE/CRITICAL\):\s*(\d+)', img_block)
        mild_match = re.search(r'MILD findings:\s*(\d+)', img_block)
        
        ai = ai_match.group(1) if ai_match else "N/A"
        votes = votes_match.group(1) if votes_match else "0"
        mild = mild_match.group(1) if mild_match else "0"
        
        print(f"FN | {img_name:<30} | AI: {ai}% | Votes: {votes} | Mild: {mild}")
