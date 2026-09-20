import re

log_content = open('../retinaguard_analysis.log', 'r').read()

blocks = re.split(r'\[\+\] \w+ (?:IMAGE|SCAN|ANALYSIS)', log_content)

for block in blocks:
    if 'VERDICT: HEALTHY' in block and 'Retinitis' in block:
        # Extract filename (it might be in the request or somewhere)
        # We can just extract the AI and Votes
        ai_match = re.search(r'AI: ([\d\.]+)%', block)
        votes_match = re.search(r'Clinical Votes \(MODERATE/CRITICAL\):\s*(\d+)', block)
        mild_match = re.search(r'MILD findings:\s*(\d+)', block)
        
        ai = ai_match.group(1) if ai_match else "N/A"
        votes = votes_match.group(1) if votes_match else "0"
        mild = mild_match.group(1) if mild_match else "0"
        
        print(f"FN | AI: {ai:>5}% | Votes: {votes} | Mild: {mild}")
