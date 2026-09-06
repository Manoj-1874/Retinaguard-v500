# ==============================================================================
# RETINAGUARD: OUT-OF-DISTRIBUTION (OOD) VULNERABILITY PROOF
# ==============================================================================
# Role in Research Paper:
# This script represents PROOF 5: Out-of-Distribution (OOD) Rejection necessity.
# It mathematically proves the catastrophic danger of deploying a "Black Box" CNN 
# with rudimentary threshold-based gatekeeping in a clinical environment.
#
# Methodology & Findings:
# An image of a dog (dog.jpg) was fed into the diagnostic pipeline. The basic 
# security gatekeeper (alidate_image) evaluated the red-channel variance, and 
# because the dog was illuminated by warm, red-dominant sunlight, the algorithm 
# falsely authenticated it as a "Valid Color Retina".
# 
# The CNN then processed the dog's fur texture through a CLAHE Red Filter and 
# hallucinated micro-pathologies, diagnosing the dog with Retinitis Pigmentosa 
# at 99.87% confidence.
#
# Conclusion:
# This empirical failure provides the definitive justification for the final 
# 10-Scanner Hybrid CDSS (app.py). A safe clinical AI must explicitly map 
# anatomical landmarks (Optic Disc, Macula, Vasculature) before allowing the 
# CNN to diagnose. Without anatomical validation, the CNN is clinically unsafe.
# ==============================================================================

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "/content/drive/MyDrive/RP_Classification_Experiment/Models/RP_Classifier_FYP.h5"
TEST_IMAGE_PATH = "/content/drive/MyDrive/dataset2/dog.jpg"

def validate_image(img):
    """
    Basic color-variance gatekeeper.
    PROVEN FLAWED: Falsely accepts warm-lit non-medical images (like a dog).
    """
    b, g, r = cv2.split(img)
    mean_r = np.mean(r)
    mean_b = np.mean(b)
    mean_g = np.mean(g)

    if np.var([mean_r, mean_g, mean_b]) < 15:
        return True, "Valid Grayscale Retina"
    if mean_r < 10: return False, "Image too dark"
    if mean_b > (mean_r * 0.9):
        return False, "High Blue Content"
    return True, "Valid Color Retina"

# [Standard Image Processing and Prediction Functions Truncated for Archive]

# TERMINAL OUTPUT ARCHIVE:
# 👁️ Scanning: dog.jpg
# ✅ Security Pass: Valid Color Retina
# ⚙️ Path: Color Logic (Standard Retina)
# 
# 🏥 FINAL FORTRESS DIAGNOSIS
# STATUS: 🔴 POSITIVE (RP DETECTED)
# CONFIDENCE: 99.87%
# REASON: Pigment confirmed (Red Filter)
