# ==============================================================================
# DATASET DEGRADATION: CATARACT / FOG SIMULATION (STRESS TEST GENERATOR)
# ==============================================================================
# Role in Research Paper:
# To rigorously evaluate the robustness of the Hybrid CDSS in real-world clinical 
# scenarios, we must test it against severely degraded inputs. One of the most 
# common causes of poor fundus image quality in elderly patients is Cataracts 
# (which manifest as severe blur and light-scattering haze).
#
# Methodology:
# This script artificially synthesizes a severe Cataract condition on a clear 
# retinal scan. 
# 1. Blur (Gaussian): Simulates the loss of high-frequency detail (lens opacity).
# 2. Haze (Alpha Blending): Simulates light scattering, drastically reducing 
#    the dynamic range and contrast of the image (40% White Haze).
#
# Academic Conclusion:
# By generating these adversarial, degraded images, we can quantitatively measure 
# how quickly the diagnostic CNN collapses under noise, and whether the CDSS 
# "Smart Metrics" (Sharpness/Contrast detection) successfully catches the fog 
# and triggers the Auto-Dehazing engine.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the target image
img_path = "/content/drive/MyDrive/dataset2/Retinitis-pigmentosa-sine-pigmento-OD_affected.jpg"  
original = cv2.imread(img_path)

if original is None:
    print("❌ Error: Could not find image. Please check the path.")
else:
    # 2. Simulate Cataract (Blur + Haze)
    
    # Blur: Simulates opacity in the lens (destroys high-frequency details)
    blurred = cv2.GaussianBlur(original, (25, 25), 0)

    # Haze: Simulates light scattering (reduces contrast/dynamic range)
    overlay = original.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (255, 255, 255), -1)
    
    # 40% White Haze Blended with the Blurred Image
    hazy = cv2.addWeighted(blurred, 0.6, overlay, 0.4, 0) 

    # 3. Save the degraded test image
    save_path = "/content/drive/MyDrive/dataset2/cataract_test_affected2.jpg"
    cv2.imwrite(save_path, hazy)
    print(f"✅ Generated Test Image: {save_path}")

    # 4. Display the degradation for the research paper figure
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title("Original Fundus")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)); plt.title("Simulated Cataract (Degraded)")
    plt.show()

# TERMINAL OUTPUT ARCHIVE:
# ✅ Generated Test Image: /content/drive/MyDrive/dataset2/cataract_test_affected2.jpg
