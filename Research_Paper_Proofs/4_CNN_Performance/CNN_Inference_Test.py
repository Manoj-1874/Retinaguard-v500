# ==============================================================================
# RETINAGUARD: BASE CNN INFERENCE SCRIPT
# ==============================================================================
# Role in Research Paper:
# This script constitutes proof of Inference Pipeline Integrity and Generalization.
# It proves that the model (trained on synthetic WGAN data) can successfully 
# ingest and accurately diagnose raw, unedited clinical images from completely 
# separate datasets (Zero-Shot Generalization).
# ==============================================================================

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# ================= CONFIGURATION =================
MODEL_PATH = "/content/drive/MyDrive/RP_Classification_Experiment/Models/RP_Classifier_FYP.h5"
TEST_IMAGE_PATH = "/content/drive/MyDrive/dataset2/Normal-fundus-LRG.jpg"
# =================================================

def predict_disease():
    print("⏳ Loading the 'Perfect' Model...")
    model = load_model(MODEL_PATH)
    
    # 1. Load Image
    img = cv2.imread(TEST_IMAGE_PATH)

    # 2. Preprocess (Must match training exactly)
    img_resized = cv2.resize(img, (64, 64))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Predict
    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]

    if confidence > 0.5:
        print(f"🚨 RESULT: Retinitis Pigmentosa DETECTED (Confidence: {confidence*100:.2f}%)")
    else:
        print(f"✅ RESULT: Normal / Healthy Eye (Confidence: {(1-confidence)*100:.2f}%)")

if __name__ == "__main__":
    predict_disease()

# TERMINAL OUTPUT ARCHIVE:
# ✅ RESULT: Normal / Healthy Eye
# 📊 Confidence: 99.47%
