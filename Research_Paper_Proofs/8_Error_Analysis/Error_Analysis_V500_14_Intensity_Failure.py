# ==============================================================================
# ERROR ANALYSIS: RETINAGUARD V500.14 (THE ABSOLUTE INTENSITY FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents another critical failure mode in the evolution of the 
# Hybrid CDSS: The danger of "Absolute Intensity Thresholding" when analyzing 
# biological pathology.
#
# The Architecture & The Trap:
# In V500.14, an "Intensity Filter" was added to the XAI Pigment Scanner to 
# reduce False Positives caused by shadows or dark choroidal vessels. 
# The logic: `if mean_grayscale_intensity > 50.0, ignore the cluster`.
# The assumption was that true Retinitis Pigmentosa bone spicules are always 
# pitch black (intensity < 50).
#
# The Failure Cascade:
# 1. The system processed a classic RP scan with massive, undeniable pigment 
#    blobs (`image-full (5)Affected.jpg`).
# 2. The morphological scanner detected hundreds of valid clusters.
# 3. However, due to camera exposure and the patient's melanin levels, the 
#    pigment blobs were dark brown/grey (intensity ~60-70), not pitch black.
# 4. The Intensity Filter triggered on EVERY SINGLE BLOB, coloring them Green 
#    (Ignored) and erasing them from the pathology map.
# 5. The Texture Score plummeted to 0.00%.
# 6. The CNN (AI) correctly output 99.98% confidence.
# 7. Because AI was 99% but Texture was 0%, the rule engine fell back to 
#    "SUSPICIOUS (Sine Pigmento)" - completely misclassifying severe, late-stage 
#    pigmentary RP as an early-stage occult variant.
#
# Academic Conclusion:
# This proves that Absolute Intensity (pixel brightness) is a fundamentally unsafe 
# metric in retinal analysis. Because biological melanin and camera exposures vary 
# wildly, pathology must be defined by morphological shape and local contrast 
# (relative difference to surrounding tissue), NEVER by hard-coded grayscale values.
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

# [RetinaGuardV500_14_Intensity Implementation Truncated for Archive]
# Full source identical to V500.14 codebase.

# TERMINAL OUTPUT ARCHIVE:
# [05:44:36] ⚙️ INITIALIZING RETINAGUARD V500.14 (INTENSITY)...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
#
# [05:44:38] 🚀 PROCESSING: image-full (5)Affected.jpg
# ✅ Quality OK. Proceeding...
#       ✨ Contrast OK (C:58.0). No Dehaze.
#       🤖 AI Confidence: 99.98%
#
# 🔬 RETINAGUARD V500.14 REPORT
# ============================================================
#  1. DIAGNOSIS:      SUSPICIOUS (Sine Pigmento)
#  2. SEVERITY:       Occult
#  3. EXPLAINABILITY: AI detected subtle features.
#  4. METRICS:        Frag: 0.0 | Texture: 0.00%
# ------------------------------------------------------------
