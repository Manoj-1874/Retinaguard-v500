import os
import cv2
import numpy as np
from app import evaluate_scan, model, preprocess_image

images = [
    'Retinitis Pigmentosa100.jpg',
    'Retinitis Pigmentosa101.jpg',
    'Retinitis Pigmentosa13.jpg',
    'Retinitis Pigmentosa134.jpg',
    'Retinitis Pigmentosa136.jpg'
]

for img in images:
    path = os.path.join(r"e:\V500\Dataset\Original Dataset\Retinitis Pigmentosa", img)
    
    # 1. Load image
    image = cv2.imread(path)
    if image is None:
        continue
        
    # 2. Preprocess
    proc = preprocess_image(image)
    proc_expanded = np.expand_dims(proc, axis=0)
    
    # 3. AI prediction
    prediction = model.predict(proc_expanded, verbose=0)
    ai_confidence = float(prediction[0][0])
    
    # 4. Evaluate scan
    result = evaluate_scan(image, ai_confidence, 35, 0.0)
    
    print(f"\n--- {img} ---")
    print(f"Verdict: {result['verdict_code']} | AI: {ai_confidence*100:.1f}%")
    
    # Extract features
    cf = result['clinical_features']
    print(f"Vessel: {cf.get('vessel_density_raw', 0):.1f}% -> Final: {cf.get('vessel_density', 0):.1f}%")
    print(f"Texture: Ent={cf.get('texture_entropy', 0):.2f}, Var={cf.get('texture_local_variation', 0):.2f}")
    print(f"Optic Disc: {cf.get('optic_disc_brightness', 0):.1f}")
