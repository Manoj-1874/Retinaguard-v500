# ==============================================================================
# DATASET DEGRADATION: HARDWARE SENSOR NOISE & UNDEREXPOSURE (STRESS TEST)
# ==============================================================================
# Role in Research Paper:
# While the previous test simulated biological degradation (Cataracts), this 
# script simulates hardware degradation. In rural or low-resource clinical 
# settings, fundus cameras often suffer from poor flash lighting and cheap, 
# grainy CMOS sensors.
#
# Methodology:
# This script applies a two-stage adversarial degradation to a clean clinical scan:
# 1. Gaussian Noise Injection: We inject stochastic normal-distribution noise 
#    across the image matrix to simulate high-ISO sensor grain.
# 2. Illumination Attenuation: We convert the image to LAB color space and 
#    multiply the Luminance (L) channel by 0.8 to simulate a 20% underexposure 
#    (weak camera flash).
#
# Academic Conclusion:
# Standard CNNs often misclassify high-frequency sensor noise as pathology 
# (False Positives) or fail entirely when the image is too dark. Generating 
# these exact stress tests proves the necessity of the system's "Smart Metrics" 
# gatekeeper, which dynamically measures contrast/sharpness to either reject 
# the image or apply calibrated enhancement before AI inference.
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a CLEAN Healthy Image
img_path = "/content/drive/MyDrive/dataset2/1_right_Healthy.jpg"
original = cv2.imread(img_path)

if original is None:
    print("❌ Error: Could not find image. Please check the path.")
else:
    # 2. Simulate "Global Standard" Poor Quality (Hardware Degradation)

    # A. Add Gaussian Noise (Simulates grainy high-ISO sensor)
    row, col, ch = original.shape
    mean = 0
    var = 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = original + gauss * 20 # Strength of noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # B. Lower Contrast (Simulates bad lighting / underexposure)
    # Convert to LAB, scale down L channel, convert back
    lab = cv2.cvtColor(noisy, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.multiply(l, 0.8) # Reduce brightness/contrast by 20%
    lab_low = cv2.merge((l, a, b))
    final_stress = cv2.cvtColor(lab_low, cv2.COLOR_LAB2BGR)

    # 3. Save the test image
    save_path = "/content/drive/MyDrive/dataset2/global_stress_test.jpg"
    cv2.imwrite(save_path, final_stress)
    print(f"✅ Generated Stress Test Image: {save_path}")

    # 4. Display the Challenge for the Research Paper Figure
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title("Original (Perfect)")
    plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(final_stress, cv2.COLOR_BGR2RGB)); plt.title("Stress Test (Noisy/Dark)")
    plt.show()

# TERMINAL OUTPUT ARCHIVE:
# ✅ Generated Stress Test Image: /content/drive/MyDrive/dataset2/global_stress_test.jpg
