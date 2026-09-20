with open(r"e:\V500\public\index.html", 'r', encoding='utf-8') as f:
    content = f.read()

# Find loadReportDetail function block
start_idx = content.find("function loadReportDetail")
if start_idx != -1:
    # Print the next 2000 characters
    print("--- loadReportDetail code ---")
    print(content[start_idx:start_idx+2000])
else:
    print("loadReportDetail not found.")
