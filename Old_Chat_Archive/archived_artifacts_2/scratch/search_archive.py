with open(r"e:\V500\public\index.html", 'r', encoding='utf-8') as f:
    content = f.read()

# Find openArchive function block
start_idx = content.find("function openArchive")
if start_idx != -1:
    # Print the next 1500 characters
    print("--- openArchive code ---")
    print(content[start_idx:start_idx+2000])
else:
    print("openArchive not found.")
