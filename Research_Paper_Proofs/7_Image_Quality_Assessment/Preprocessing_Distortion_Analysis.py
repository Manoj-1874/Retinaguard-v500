# ==============================================================================
# IMAGE QUALITY ASSESSMENT: PREPROCESSING DISTORTION & ROI MASKING
# ==============================================================================
# Role in Research Paper:
# This script documents a critical source of False Positives in clinical AI: 
# Preprocessing Damage. 
#
# The Problem:
# Standard image enhancement techniques (like CLAHE/Dehazing) used to prepare 
# images for CNNs can inadvertently hallucinate pathology. Specifically, at the 
# hard black border of the circular retinal scan, contrast enhancement creates 
# a jagged, noisy edge that the CNN or Morphological Scanner misinterprets as 
# peripheral Retinitis Pigmentosa.
#
# The Solution:
# The script demonstrates the "ROI Fix": calculating the center of the image 
# and applying a circular bitwise mask at 95% radius (`int(rows/2 * 0.95)`). 
# This mathematically crops out the outer 5% of the image border, deleting the 
# preprocessing artifacts before they can be analyzed by the diagnostic engine.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_preprocessing_damage(image_path):
    # 1. Load Original
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 2. Simulate the "Dehaze" (The likely culprit)
    # Simple dehaze simulation: Inverting -> Equalizing -> Inverting
    b, g, r = cv2.split(original)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_g = clahe.apply(g)
    
    # This creates a "harsh" version often confused with dehazing in medical code
    simulated_input = cv2.merge((b, enhanced_g, r))

    # 3. Create the ROI Mask (The Fix)
    mask = np.zeros_like(g)
    rows, cols = mask.shape
    # Draw a white circle in the center (ignoring corners)
    cv2.circle(mask, (cols//2, rows//2), int(rows/2 * 0.95), 255, -1)

    # 4. Visualize
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("1. What Human Sees (Healthy)")
    plt.imshow(original)

    plt.subplot(1, 3, 2)
    plt.title("2. What Model Likely Sees\n(High Contrast/Dehaze Artifacts)")
    plt.imshow(simulated_input)

    plt.subplot(1, 3, 3)
    plt.title("3. The ROI Fix\n(Masking out the sharp edges)")
    plt.imshow(cv2.bitwise_and(original, original, mask=mask))

    plt.show()

if __name__ == "__main__":
    # Test on a healthy image to visualize preprocessing damage
    check_preprocessing_damage('/content/drive/MyDrive/dataset2/1_right_Healthy.jpg')
