import cv2
import numpy as np

img = cv2.imread(r"e:\V500\uploads\Fluorescein_Angiography_Test.jpg")
if img is None:
    print("Failed to load image")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

saturation = hsv[:, :, 1]
mean_saturation = np.mean(saturation)

b, g, r = cv2.split(img)
rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))
max_channel_diff = max(rg_diff, rb_diff, gb_diff)

std_brightness = np.std(gray)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
dark_pixels = np.sum(hist[0:80])
bright_pixels = np.sum(hist[180:256])
total_pixels = img.shape[0] * img.shape[1]
dark_ratio = dark_pixels / total_pixels

print(f"Mean Saturation: {mean_saturation:.4f}")
print(f"Max Channel Diff: {max_channel_diff:.4f}")
print(f"Std Brightness: {std_brightness:.4f}")
print(f"Dark Ratio: {dark_ratio:.4f}")

score = 0
reasons = []

if mean_saturation < 20:
    score += 3
    reasons.append("Low color saturation")
elif mean_saturation < 35:
    score += 1
    reasons.append("Reduced saturation")

if max_channel_diff < 5:
    score += 3
    reasons.append("Grayscale image")
elif max_channel_diff < 15:
    score += 1
    reasons.append("Near-grayscale")

if std_brightness > 60:
    score += 2
    reasons.append("High contrast")

if dark_ratio > 0.5:
    score += 2
    reasons.append("Predominantly dark")

print(f"Score: {score}")
print(f"Reasons: {reasons}")
print(f"Is Angiography: {score >= 5}")
