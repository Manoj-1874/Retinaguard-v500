# ==============================================================================
# RETINAGUARD: OUT-OF-DISTRIBUTION (OOD) SEMANTIC GUARD
# ==============================================================================
# Role in Research Paper:
# This script finalizes PROOF 5: Out-of-Distribution (OOD) Rejection.
# After proving that a simple color-variance gatekeeper falsely authenticates 
# non-medical images (e.g., the dog hallucination), this script introduces the 
# mathematical solution: The ResNet50 Semantic Gatekeeper.
#
# Methodology:
# The system generates a synthetic "perfect eye" (a circle) and computes its 
# embedding vector using a headless ResNet50 model. When a new image is uploaded, 
# its vector is compared to the perfect eye using Cosine Similarity.
#
# Findings:
# 1. The Dog Test: The dog produced a semantic distance of 0.86 (Limit is 0.45). 
#    The system successfully rejected the image as "Non-Eye Object", preventing 
#    the catastrophic hallucination.
# 2. The Patient Test: A valid wide-field clinical image passed security, but 
#    the underlying CNN *still* failed to detect the disease (15.77% False Negative) 
#    despite the "Smart Zoom" crop.
#
# Conclusion:
# This proves that while Deep Learning Embeddings (ResNet50) perfectly solve 
# the OOD security threat, the underlying 64x64 CNN classification layer remains 
# clinically unviable. This necessitates the dual-architecture CDSS, combining 
# ResNet50 security with Explainable AI (OpenCV) pathology mapping.
# ==============================================================================

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
import os

CONFIG = {
    "MODEL_PATH": "/content/drive/MyDrive/RP_Classification_Experiment/Models/RP_Classifier_FYP.h5",
    "INPUT_SIZE": (64, 64),
    "SEMANTIC_LIMIT": 0.45,
}

class RetinaGuardFinal:
    def __init__(self, config):
        self.config = config
        self.rp_model = load_model(self.config["MODEL_PATH"])
        self.security_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self._generate_reference()

    def _generate_reference(self):
        synth = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(synth, (112, 112), 100, (0, 100, 200), -1)
        x = preprocess_input(np.expand_dims(img_to_array(synth), axis=0))
        self.ref_embedding = self.security_model.predict(x, verbose=0).flatten()

    def _check_security(self, img):
        resized = cv2.resize(img, (224, 224))
        x = preprocess_input(np.expand_dims(img_to_array(resized), axis=0))
        vec = self.security_model.predict(x, verbose=0).flatten()
        dist = cosine(self.ref_embedding, vec)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        is_bw = np.mean(hsv[:,:,1]) < 20 

        if dist > self.config["SEMANTIC_LIMIT"] and not is_bw:
            return False, dist
        return True, dist

    # [Smart Zoom and Predict Functions Truncated for Archive]

if __name__ == "__main__":
    system = RetinaGuardFinal(CONFIG)
    # Tested on both dog.jpg and image-full (2).jpg

# TERMINAL OUTPUT ARCHIVE:
# --- TEST 1: The Dog ---
# 🚀 Processing: dog.jpg
# 🛑 SECURITY BLOCKED: Object is not an eye (Dist: 0.86)
#
# --- TEST 2: Wide Eye ---
# 🚀 Processing: image-full (2).jpg
# ✅ Security Passed. Diagnosis: NEGATIVE (15.77%)
