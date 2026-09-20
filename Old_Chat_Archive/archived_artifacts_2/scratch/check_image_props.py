import cv2

img = cv2.imread(r"e:\V500\uploads\Sectoral_RP_Test.jpg")
if img is None:
    print("Image loaded as None!")
else:
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
