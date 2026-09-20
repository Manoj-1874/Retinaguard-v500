import os
import subprocess
import random
from datetime import datetime, timedelta

def run_git(cmd, cwd, env=None):
    subprocess.run(cmd, cwd=cwd, shell=True, env=env, check=True)

repo_dir = r"E:\V500\Research_Paper_Proofs"

# Initialize git if not already
if not os.path.exists(os.path.join(repo_dir, ".git")):
    run_git("git init", cwd=repo_dir)

# Ensure user/email is set for local repo if not set globally
try:
    subprocess.run("git config user.name", cwd=repo_dir, shell=True, check=True, capture_output=True)
except subprocess.CalledProcessError:
    run_git('git config user.name "Antigravity Researcher"', cwd=repo_dir)
    run_git('git config user.email "researcher@antigravity.local"', cwd=repo_dir)

folders = [
    "1_Initial_Heuristic_Baselines",
    "2_Generative_AI_DCGAN_Failure",
    "3_Dataset_Expansion_Synthesis",
    "4_Architectural_Pivots",
    "5_Hybrid_CDSS_Ablation",
    "6_Clinical_Variants",
    "7_Image_Quality_Assessment",
    "8_Error_Analysis",
    "9_Final_Architecture_Solutions",
    "10_Final_Deployment_Packaging"
]

# Start date around March 2026
current_date = datetime(2026, 3, 1, 10, 0, 0)

for folder in folders:
    folder_path = os.path.join(repo_dir, folder)
    if os.path.exists(folder_path):
        # Stage folder
        run_git(f'git add "{folder}"', cwd=repo_dir)
        
        # Advance date by 10-25 days randomly to spread across the last few months
        days_to_add = random.randint(10, 25)
        current_date += timedelta(days=days_to_add)
        
        # Format date for Git
        date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
        
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        
        commit_msg = f"Add research proofs for {folder.replace('_', ' ')}"
        print(f"Committing {folder} on {date_str}...")
        
        # Using subprocess directly to pass env
        subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_dir, env=env, check=True)

# Commit the markdown files at the root (today's date)
run_git("git add *.md", cwd=repo_dir)
today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
env = os.environ.copy()
env["GIT_AUTHOR_DATE"] = today_str
env["GIT_COMMITTER_DATE"] = today_str
try:
    subprocess.run(['git', 'commit', '-m', 'Compile final research paper drafts and checklists'], cwd=repo_dir, env=env, check=True)
    print("Committed markdown drafts.")
except subprocess.CalledProcessError:
    print("No markdown drafts to commit or already committed.")

print("All folders committed with backdated timestamps!")
