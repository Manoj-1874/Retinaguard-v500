import sys
sys.path.append(r"e:\V500")

import cv2
from image_quality_validator import validate_image_quality

img = cv2.imread(r"e:\V500\uploads\Sectoral_RP_Test.jpg")
if img is None:
    print("Failed to load image")
    exit(1)

print("Loaded image shape:", img.shape)
try:
    res = validate_image_quality(img, "TEST-SEC")
    print("Success! Result:")
    print("Valid:", res['valid'])
    print("Quality Score:", res['quality_score'])
    print("Warnings:", res['warnings'])
    print("Errors:", res['errors'])
except Exception as e:
    print("Failed with exception:", e)
    import traceback
    traceback.print_exc()
