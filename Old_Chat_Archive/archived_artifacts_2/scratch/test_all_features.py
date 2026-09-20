import sys
sys.path.append(r"e:\V500")

import cv2
import numpy as np

# Load image
img = cv2.imread(r"e:\V500\uploads\Sectoral_RP_Test.jpg")
if img is None:
    print("Failed to load image")
    exit(1)

print("Loaded image shape:", img.shape)

# Import functions from app
try:
    from app import (
        get_fov_mask,
        detect_angiography,
        extract_vessel_features,
        extract_pigment_features,
        extract_optic_disc_features,
        extract_texture_features,
        extract_quadrant_features
    )
    print("Successfully imported features from app.py")
except Exception as e:
    print("Failed to import from app.py:", e)
    import traceback
    traceback.print_exc()
    exit(1)

# Step-by-step execution to locate the cvtColor error
try:
    print("\n--- Step 1: get_fov_mask ---")
    fov_mask = get_fov_mask(img)
    print("fov_mask shape:", fov_mask.shape)
    
    print("\n--- Step 2: detect_angiography ---")
    is_angio, angio_confidence, angio_reason = detect_angiography(img)
    print(f"is_angio: {is_angio}, reason: {angio_reason}")
    
    print("\n--- Step 3: extract_vessel_features ---")
    vessel_feats = extract_vessel_features(img, fov_mask, is_angiography=is_angio)
    print("vessel_feats keys:", list(vessel_feats.keys()))
    
    print("\n--- Step 4: extract_pigment_features ---")
    pigment_feats = extract_pigment_features(img, fov_mask)
    print("pigment_feats keys:", list(pigment_feats.keys()))
    
    print("\n--- Step 5: extract_optic_disc_features ---")
    optic_disc_feats = extract_optic_disc_features(img, fov_mask)
    print("optic_disc_feats keys:", list(optic_disc_feats.keys()))
    
    print("\n--- Step 6: extract_texture_features ---")
    texture_feats = extract_texture_features(img, fov_mask)
    print("texture_feats keys:", list(texture_feats.keys()))
    
    print("\n--- Step 7: extract_quadrant_features ---")
    quadrant_feats = extract_quadrant_features(img, fov_mask)
    print("quadrant_feats keys:", list(quadrant_feats.keys()))
    
    print("\n--- All steps completed successfully! ---")
    
except Exception as e:
    print("\n[FAILED] Encountered error:")
    print(e)
    import traceback
    traceback.print_exc()
