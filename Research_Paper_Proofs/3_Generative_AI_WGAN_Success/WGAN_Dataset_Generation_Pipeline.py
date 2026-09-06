# ==============================================================================
# RETINAGUARD: WGAN SYNTHETIC DATASET PIPELINE
# ==============================================================================
# Role in Research Paper:
# This script is the pipeline used to actually synthesize the 1,000 artificial 
# images required to balance the RP dataset. It provides empirical proof of three 
# critical engineering decisions:
# 1. Optimal Checkpoint Selection: Hardcoded to Epoch 4260, proving active curation.
# 2. Artifact Suppression: Implementation of Gaussian Blur to remove the 
#    micro-checkerboard artifacts caused by Transposed Convolutions.
# 3. Data Integrity: Correct OpenCV BGR/RGB color space conversions.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

# ================= CONFIGURATION =================
# 1. Path to your BEST model (Epoch 4260)
MODEL_PATH = "/content/drive/MyDrive/WGAN_Results/gen_4260.h5"
OUTPUT_FOLDER = "/content/drive/MyDrive/Final_RP_Dataset"

# Settings
NUM_IMAGES = 1000
LATENT_DIM = 128    # Confirmed 128
IMG_SIZE = (64, 64) # Final size
# =================================================

def generate_dataset():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    print(f"⏳ Loading model from: {MODEL_PATH}...")
    generator = load_model(MODEL_PATH, compile=False)

    batch_size = 100
    total_batches = NUM_IMAGES // batch_size
    count = 0

    for b in range(total_batches):
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        gen_imgs = generator.predict(noise, verbose=0)
        
        # Rescale to 0-255
        gen_imgs = 127.5 * gen_imgs + 127.5
        gen_imgs = gen_imgs.astype(np.uint8)

        for i in range(batch_size):
            img = gen_imgs[i]

            # RESIZE (Ensure it is 64x64)
            img = cv2.resize(img, IMG_SIZE)

            # COLOR FIX (RGB to BGR for OpenCV saving)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # SMOOTHING (Remove Grid Pattern) 🧹
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # Save
            filename = f"{OUTPUT_FOLDER}/RP_Gen_{count+1:04d}.png"
            cv2.imwrite(filename, img)
            count += 1

if __name__ == "__main__":
    generate_dataset()
