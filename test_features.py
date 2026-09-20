import os
import cv2
import numpy as np

import app

images = [
    'Retinitis Pigmentosa13.jpg',
    'Retinitis Pigmentosa134.jpg',
    'Retinitis Pigmentosa136.jpg'
]

for img in images:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img)
    image = cv2.imread(path)
    if image is None: continue
    
    fov_mask = app.get_fov_mask(image)
    
    vessel = app.extract_vessel_features(image, fov_mask)
    texture = app.extract_texture_features(image, fov_mask)
    optic = app.extract_optic_disc_features(image, fov_mask)
    
    print(f"\n--- {img} ---")
    print(f"Vessel: {vessel}")
    print(f"Texture: {texture}")
    print(f"Optic: {optic}")
