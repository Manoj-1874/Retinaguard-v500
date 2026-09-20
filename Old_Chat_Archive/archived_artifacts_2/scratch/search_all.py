import os

search_terms = ["optos", "widefield"]
workspace_dir = r"e:\V500"

for root, dirs, files in os.walk(workspace_dir):
    for file in files:
        if file.endswith('.py') or file.endswith('.js') or file.endswith('.html'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    for term in search_terms:
                        if term.lower() in line.lower():
                            safe_line = line.strip().encode('ascii', 'backslashreplace').decode('ascii')
                            print(f"{file}:{i} ({term}): {safe_line}")
            except Exception as e:
                pass
