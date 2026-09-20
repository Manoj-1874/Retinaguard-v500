import cv2
import numpy as np

img = cv2.imread(r"e:\V500\uploads\Fluorescein_Angiography_Test.jpg")
h, w, c = img.shape
cy, cx = h // 2, w // 2

# Print a 5x5 window of BGR values around center
print("Center 5x5 BGR pixels:")
for y in range(cy-2, cy+3):
    row = []
    for x in range(cx-2, cx+3):
        row.append(list(img[y, x]))
    print(row)
