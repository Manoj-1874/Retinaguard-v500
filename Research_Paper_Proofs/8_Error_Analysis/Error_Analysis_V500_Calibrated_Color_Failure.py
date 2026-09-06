# ==============================================================================
# ERROR ANALYSIS: RETINAGUARD V500 CALIBRATED (THE COLOR FILTER FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents the final major failure mode encountered in the Hybrid CDSS 
# architecture: The danger of Color-Based Thresholding (The False Negative Trap).
#
# The Architecture:
# To solve the Tigroid False Positives encountered in V145 and V500, a "Calibrated" 
# preprocessing engine was built. If an image had high texture energy (>100), the 
# system entered "SAFE MODE" to prevent over-enhancing healthy stripes.
# In SAFE MODE, it applied a strict Color Filter to every detected pigment cluster:
# If Red > Green * 1.15, the cluster was assumed to be a healthy blood vessel 
# or Tigroid stripe, and was subsequently IGNORED.
#
# The Failure Cascade:
# 1. The system was fed a Montage scan with severe Retinitis Pigmentosa (`image-full (3)Affected.jpg`).
# 2. Because the image was highly textured, the system entered SAFE MODE.
# 3. The scanner correctly detected hundreds of bone spicule clusters.
# 4. However, because ambient fundus lighting gives the entire retina (including 
#    the pigment clumps) a reddish hue, the RP bone spicules passed the `Red > Green * 1.15` test.
# 5. The system explicitly labeled almost every single bone spicule as a "Healthy Vessel" 
#    (marked in Green on the debug map) and erased them from the pathology map.
# 6. The Texture Score plummeted to 0.49%, resulting in a Catastrophic False Negative (HEALTHY).
#
# Academic Conclusion:
# This failure proves that absolute color thresholding is fundamentally unsafe in 
# fundus imaging due to extreme variations in ambient illumination and melanin 
# concentration across patients. It proves that morphological properties (shape, 
# density, fragmentation) are the ONLY mathematically safe way to classify retinal 
# pigment, not raw RGB ratios.
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

# [RetinaGuardV500_Calibrated Implementation Truncated for Archive]
# Full source identical to V500 Calibrated codebase.

# TERMINAL OUTPUT ARCHIVE:
# [14:40:18] ⚙️ INITIALIZING RETINAGUARD V500 (CALIBRATED)...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
#
# [14:40:19] 🚀 PROCESSING: image-full (3)Affected.jpg
# ✅ Quality OK. Proceeding...
#       🤖 AI Confidence: 0.00%
#       🩺 PREPROCESS: SAFE (Red/Tigroid Detected). Energy: 232.8
#
# 🔬 RETINAGUARD V500 CALIBRATED REPORT
# ============================================================
#  1. DIAGNOSIS:      HEALTHY
#  2. SEVERITY:       None
#  3. EXPLAINABILITY: Normal Fundus Appearance
#  4. METRICS:        Frag: 39.8 | Texture: 0.49%
# ------------------------------------------------------------
