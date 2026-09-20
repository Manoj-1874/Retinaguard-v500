import cv2
import numpy as np

img = cv2.imread(r"e:\V500\uploads\Fluorescein_Angiography_Test.jpg")
if img is None:
    print("Failed to load image")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

# Calculate chroma (max - min)
max_ch = np.max(img, axis=2)
min_ch = np.min(img, axis=2)
chroma = max_ch - min_ch

masked_chroma = chroma[mask > 0]
mean_chroma = np.mean(masked_chroma) if len(masked_chroma) > 0 else 0

print(f"Retinal Mask Mean Chroma: {mean_chroma:.4f}")
