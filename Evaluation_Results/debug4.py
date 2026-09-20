import re

log = open('../retinaguard_analysis.log', 'r', encoding='utf-8', errors='ignore').read()
scans = log.split('======================================================================')

fn_count = 0
for scan in scans:
    if 'VERDICT: HEALTHY' in scan and 'Retinitis' in scan:
        name_match = re.search(r'Image:\s*(.*?\.jpg)', scan)
        name = name_match.group(1).split('\\')[-1].split('/')[-1] if name_match else "Unknown"
        
        ai_match = re.search(r'AI:\s*([\d\.]+)%', scan)
        votes_match = re.search(r'Clinical Votes \(MODERATE/CRITICAL\):\s*(\d+)', scan)
        mild_match = re.search(r'MILD findings:\s*(\d+)', scan)
        
        ai = ai_match.group(1) if ai_match else "N/A"
        votes = votes_match.group(1) if votes_match else "0"
        mild = mild_match.group(1) if mild_match else "0"
        
        if name != "Unknown":
            print(f"{name:30} | AI: {ai:>5}% | Votes: {votes} | Mild: {mild}")
            fn_count += 1

print(f"Total FNs found in log: {fn_count}")
