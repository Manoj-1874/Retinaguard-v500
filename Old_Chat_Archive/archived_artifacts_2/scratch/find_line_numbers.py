html_path = r"e:\V500\public\index.html"

with open(html_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "function loadReportDetail" in line:
        print(f"Found 'function loadReportDetail' on line {idx+1}")
        # Print surrounding lines
        for j in range(max(0, idx-2), min(len(lines), idx+5)):
            print(f"{j+1:4d}: {lines[j].rstrip()}")
        break
