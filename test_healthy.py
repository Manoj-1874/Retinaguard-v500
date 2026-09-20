import os
import cv2
import app

images = [
    'Healthy148.jpg',
    'Healthy163.jpg',
    'Healthy177.jpg',
    'Healthy183.jpg'
]

for img in images:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Healthy", img)
    image = cv2.imread(path)
    if image is None: continue
    
    fov_mask = app.get_fov_mask(image)
    
    vessel = app.extract_vessel_features(image, fov_mask)
    texture = app.extract_texture_features(image, fov_mask)
    optic = app.extract_optic_disc_features(image, fov_mask)
    pigment = app.extract_pigment_features(image, fov_mask)
    
    print(f"\n--- {img} ---")
    print(f"Vessel: Density {vessel.get('density', 0)*100:.1f}%")
    print(f"Texture: Entropy {texture.get('entropy', 0):.2f}, Var {texture.get('local_variation', 0):.2f}")
    print(f"Optic: Brightness {optic.get('disc_brightness', 0):.1f}")
    print(f"Pigment: Clusters {pigment.get('num_clusters', 0)}")
