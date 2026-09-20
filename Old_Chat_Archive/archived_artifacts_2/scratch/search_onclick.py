with open(r"e:\V500\public\index.html", 'r', encoding='utf-8') as f:
    content = f.read()

# Let's search for "crate-card" or "crates" in scripts
script_content = content[content.find("<script>"):]
print("Occurrences of 'crate-card' in script section:")
for idx, line in enumerate(script_content.split('\n')):
    if "crate-card" in line or "crate.onclick" in line or "addEventListener" in line:
        print(f"Line {idx+1}: {line.strip()}")
        
# Find if there are any other places where detail-modal is opened
print("\nOccurrences of 'detail-modal' in script section:")
for idx, line in enumerate(script_content.split('\n')):
    if "detail-modal" in line:
        print(f"Line {idx+1}: {line.strip()}")
