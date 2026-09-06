# ==============================================================================
# ERROR ANALYSIS: RETINAGUARD V500.15 (THE DUAL-PATH FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents the final failed iteration before the architectural 
# breakthrough of V500.10. It proves that combining multiple flawed heuristics 
# (Color Thresholds + Absolute Intensity Thresholds) does not fix the underlying 
# issue; it merely creates a more restrictive filter that exacerbates False Negatives.
#
# The Architecture & The Trap:
# In V500.15, the "Dual-Path Doctor Logic" was implemented. For color images, 
# it checked every detected cluster against two hard-coded rules:
# 1. Is it Red? (Red > Green * 1.15) -> Assume healthy vessel.
# 2. Is it dark enough? (Intensity > 48) -> Assume healthy shadow.
#
# The Failure Cascade:
# 1. The system processed the severe RP montage (`image-full (3)Affected.jpg`).
# 2. Because the ambient fundus camera flash gives the entire retina (and the 
#    spicules) a reddish hue, the pigment passed the "Is it Red?" test.
# 3. The algorithm immediately labeled massive, undeniable bone spicules as 
#    "Healthy Vessels" (colored Green in Panel 3), completely ignoring their 
#    pathological shape and density.
# 4. Texture score dropped to 0.00%, resulting in a Catastrophic False Negative 
#    (HEALTHY) on a severely diseased patient.
#
# Academic Conclusion:
# This is the definitive proof that Explainable AI (XAI) in ophthalmology cannot 
# rely on absolute pixel values (RGB or Grayscale Intensity). Biological variance 
# and hardware illumination make these thresholds inherently unstable. This failure 
# directly justified the creation of the V500.10 "Jaggedness Filter," which 
# abandoned pixel thresholds entirely in favor of geometric shape analysis 
# (Aspect Ratio & Convex Hull Solidity).
# ==============================================================================

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
import os
import glob
from datetime import datetime

# [RetinaGuardV500_15_DualPath Implementation Truncated for Archive]
# Full source identical to V500.15 codebase.

# TERMINAL OUTPUT ARCHIVE:
# [05:47:09] ⚙️ INITIALIZING RETINAGUARD V500.15 (DUAL-PATH)...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
#
# [05:47:10] 🚀 PROCESSING: image-full (3)Affected.jpg
# ✅ Quality OK. Proceeding...
#       ✨ Contrast OK (C:50.3). No Dehaze.
#       🤖 AI Confidence: 0.00%
#
# 🔬 RETINAGUARD V500.15 REPORT
# ============================================================
#  1. DIAGNOSIS:      HEALTHY
#  2. SEVERITY:       None
#  3. EXPLAINABILITY: Normal Fundus Appearance
#  4. METRICS:        Frag: 0.0 | Texture: 0.00%
# ------------------------------------------------------------
