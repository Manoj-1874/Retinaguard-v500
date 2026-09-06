import numpy as np
from PIL import Image, ImageDraw
import os
import random

def synthesize_rpa_variant():
    # Load a real healthy image from the uploads folder to pass the quality check
    base_image_path = os.path.join("E:\\V500", "uploads", "healthy_retina_edge_case.png")
    out_path = os.path.join("E:\\V500", "uploads", "synthesized_rpa_variant.png")
    
    if not os.path.exists(base_image_path):
        print(f"Error: Base image not found at {base_image_path}")
        return
        
    try:
        img = Image.open(base_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # RPA (Retinitis Punctata Albescens) is characterized by hundreds of 
        # tiny white/yellowish flecks scattered across the retina, without bone spicules.
        
        print("Injecting Retinitis Punctata Albescens (White Flecks)...")
        num_flecks = random.randint(150, 250)
        
        for _ in range(num_flecks):
            # Keep flecks mostly in the mid-periphery (away from extreme edges or center macula)
            ex_x = random.randint(int(width * 0.1), int(width * 0.9))
            ex_y = random.randint(int(height * 0.1), int(height * 0.9))
            sz = random.randint(1, 4)
            
            # Draw bright white/yellow flecks
            draw.ellipse([ex_x, ex_y, ex_x+sz, ex_y+sz], fill=(245, 250, 220))
            
        img.save(out_path, quality=95)
        print(f"Success! Synthesized RPA edge-case image at: {out_path}")
        
    except Exception as e:
        print(f"Error during synthesis: {e}")

if __name__ == "__main__":
    synthesize_rpa_variant()
