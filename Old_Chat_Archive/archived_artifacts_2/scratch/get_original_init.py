import subprocess
import re

def main():
    # Retrieve the file from commit e13288f~1
    content = subprocess.check_output(["git", "show", "e13288f~1:public/index.html"], cwd="e:\\V500").decode("utf-8", errors="ignore")
    
    # Use regex or simple search to find function init()
    start_idx = content.find("function init()")
    if start_idx == -1:
        print("Could not find function init()")
        return
        
    # Find the closing brace of function init() or print next 120 lines
    print("--- OLD INIT FUNCTION IN INDEX.HTML ---")
    lines = content[start_idx:].splitlines()
    for i in range(min(120, len(lines))):
        print(lines[i])

if __name__ == "__main__":
    main()
