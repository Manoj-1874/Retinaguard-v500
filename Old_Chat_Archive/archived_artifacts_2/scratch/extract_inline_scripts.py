import re

html_path = r"e:\V500\public\index.html"
output_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\51e22eee-ff09-435e-9d76-0aac0bee55c7\scratch\inline_scripts.js"

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all <script>...</script> blocks
# Let's find script tags. Since the last script block is usually the main logic, let's extract all script contents.
script_blocks = re.findall(r'<script>(.*?)</script>', content, re.DOTALL)

with open(output_path, 'w', encoding='utf-8') as out:
    for idx, block in enumerate(script_blocks):
        out.write(f"\n// ==========================================\n")
        out.write(f"// SCRIPT BLOCK {idx+1}\n")
        out.write(f"// ==========================================\n\n")
        out.write(block)
        out.write("\n\n")

print(f"Extracted {len(script_blocks)} script blocks to {output_path}")
