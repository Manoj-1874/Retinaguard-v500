import pandas as pd
import os
import shutil

# Paths
excel_path = 'E:/RetinaGaurd_Prroject/ODIR-5K/data.xlsx'
images_dir = 'E:/RetinaGaurd_Prroject/ODIR-5K/Training Images'
dest_dir = 'E:/V500/unseen_test_data'
dest_rp = os.path.join(dest_dir, 'RP')
dest_healthy = os.path.join(dest_dir, 'Healthy')

os.makedirs(dest_rp, exist_ok=True)
os.makedirs(dest_healthy, exist_ok=True)

df = pd.read_excel(excel_path)

# Find RP cases
rp_cases = df[df['Left-Diagnostic Keywords'].str.contains('retinitis pigmentosa', case=False, na=False) | 
              df['Right-Diagnostic Keywords'].str.contains('retinitis pigmentosa', case=False, na=False)]

rp_count = 0
for idx, row in rp_cases.iterrows():
    # Left eye
    if 'retinitis pigmentosa' in str(row['Left-Diagnostic Keywords']).lower():
        src = os.path.join(images_dir, row['Left-Fundus'])
        if os.path.exists(src):
            shutil.copy(src, dest_rp)
            rp_count += 1
            
    # Right eye
    if 'retinitis pigmentosa' in str(row['Right-Diagnostic Keywords']).lower():
        src = os.path.join(images_dir, row['Right-Fundus'])
        if os.path.exists(src):
            shutil.copy(src, dest_rp)
            rp_count += 1

# Find Healthy cases
healthy_cases = df[df['N'] == 1].head(50)  # Take 50 healthy cases
healthy_count = 0
for idx, row in healthy_cases.iterrows():
    src_l = os.path.join(images_dir, row['Left-Fundus'])
    if os.path.exists(src_l):
        shutil.copy(src_l, dest_healthy)
        healthy_count += 1
    
    src_r = os.path.join(images_dir, row['Right-Fundus'])
    if os.path.exists(src_r):
        shutil.copy(src_r, dest_healthy)
        healthy_count += 1

print(f"Copied {rp_count} RP images to {dest_rp}")
print(f"Copied {healthy_count} Healthy images to {dest_healthy}")
