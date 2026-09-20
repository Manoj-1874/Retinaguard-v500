html_path = r"e:\V500\public\index.html"
output_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\display_report_detail.js"

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find("function displayReportDetail")
if start_idx != -1:
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(content[start_idx:start_idx+3500]) # write 3500 chars
    print(f"Written function to {output_path}")
else:
    print("Function not found.")
