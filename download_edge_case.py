import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import os

def create_synthetic_edge_case():
    out_path = os.path.join("E:\\V500", "uploads", "synthetic_edge_case.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. Base Canvas (Dark Orange Retina)
    width, height = 800, 800
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:] = [15, 60, 180] # BGR format? No, PIL uses RGB. So RGB: [180, 60, 15]
    img = Image.new('RGB', (width, height), color=(180, 60, 15))
    draw = ImageDraw.Draw(img)
    
    # 2. Add Vignetting (Black corners)
    # Using radial gradient logic
    for x in range(width):
        for y in range(height):
            dist = np.sqrt((x - width/2)**2 + (y - height/2)**2)
            if dist > 350:
                img.putpixel((x, y), (0, 0, 0))
            elif dist > 300:
                ratio = (350 - dist) / 50.0
                r, g, b = img.getpixel((x, y))
                img.putpixel((x, y), (int(r * ratio), int(g * ratio), int(b * ratio)))

    # 3. Add Optic Disc (Bright yellow-white circle on the right side)
    draw.ellipse([550, 350, 700, 500], fill=(240, 220, 150))
    
    # 4. Add Macula (Dark spot in the center)
    draw.ellipse([350, 380, 450, 480], fill=(120, 30, 5))
    
    # 5. Add Blood Vessels (Dark red lines originating from Optic Disc)
    # Superior arcades
    draw.line([625, 425, 400, 200, 200, 150], fill=(90, 10, 0), width=8, joint="curve")
    draw.line([625, 425, 300, 100], fill=(70, 5, 0), width=5)
    # Inferior arcades
    draw.line([625, 425, 450, 650, 250, 700], fill=(90, 10, 0), width=9, joint="curve")
    draw.line([625, 425, 350, 750], fill=(70, 5, 0), width=6)
    
    # 6. EDGE CASE ARTIFACTS: Diabetic Retinopathy Hard Exudates (Bright yellow spots)
    # This will heavily test the Differential Diagnosis module to see if it correctly overrides!
    import random
    for _ in range(40):
        ex_x = random.randint(200, 600)
        ex_y = random.randint(200, 600)
        sz = random.randint(3, 8)
        draw.ellipse([ex_x, ex_y, ex_x+sz, ex_y+sz], fill=(255, 255, 180))
        
    # 7. Apply Blur and Noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Save Image
    img.save(out_path, quality=95)
    print(f"Success! Synthesized complex DR edge-case image at: {out_path}")

if __name__ == "__main__":
    create_synthetic_edge_case()
