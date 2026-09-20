import subprocess

def extract_3d_section(commit_or_path):
    if ":" in commit_or_path:
        content = subprocess.check_output(["git", "show", commit_or_path], cwd="e:\\V500").decode("utf-8", errors="ignore")
    else:
        with open(commit_or_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
    start = content.find("// --- 3D ENGINE ---")
    if start == -1:
        start = content.find("const PARTICLE_COUNT")
    if start == -1:
        return "Not found"
        
    # Get next 150 lines
    lines = content[start:].splitlines()
    return "\n".join(lines[:120])

def main():
    old_3d = extract_3d_section("3083716:public/index.html")
    current_3d = extract_3d_section("e:\\V500\\public\\index.html")
    
    print("=== OLD 3D ENGINE SETUP (3083716) ===")
    print(old_3d)
    print("\n\n=== CURRENT 3D ENGINE SETUP ===")
    print(current_3d)

if __name__ == "__main__":
    main()
