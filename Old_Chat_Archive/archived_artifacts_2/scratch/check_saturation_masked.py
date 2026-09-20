import cv2
import numpy as np

img = cv2.imread(r"e:\V500\uploads\Fluorescein_Angiography_Test.jpg")
if img is None:
    print("Failed to load image")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create mask for retinal region (brightness > 15)
_, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

# Calculate saturation inside mask
saturation = hsv[:, :, 1]
masked_saturation = saturation[mask > 0]
mean_masked_saturation = np.mean(masked_saturation) if len(masked_saturation) > 0 else 0

print(f"Overall Mean Saturation: {np.mean(saturation):.4f}")
print(f"Retinal Mask Mean Saturation: {mean_masked_saturation:.4f}")
