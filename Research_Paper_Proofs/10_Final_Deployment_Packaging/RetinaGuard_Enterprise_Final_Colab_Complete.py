# ==============================================================================
# FINAL DEPLOYMENT PACKAGING: RETINAGUARD V500 (FINAL COLAB COMPLETE)
# ==============================================================================
# Role in Research Paper:
# This script is the absolute, definitive end of the Google Colab research notebook. 
# It merges every single heuristic hack—including the Diamond Logic Veto and a 
# custom Grayscale detection patch—into the final Enterprise pipeline.
#
# The Architecture:
# The system now attempts to dynamically adjust its logic based on modality:
# - It uses the Diamond Logic Veto (1% - 15%) to suppress Tigroid noise.
# - It checks if the image is Grayscale (`_is_grayscale`).
# - If Grayscale, it changes the Pigment Override Limit to 40.0% to prevent 
#   the Red Channel filter from hallucinating pigment.
#
# The Final Failure (The Modality Collapse):
# The researcher tested this "perfect" pipeline on `rp_10_sine.jpg`. 
# "Sine Pigmento" is a rare form of RP that presents with almost ZERO dark pigment.
# However, because this was a Grayscale FAF scan, the Red Channel thresholding 
# completely collapsed. It registered the natural dark macula and vessels as 
# massive pigment clumps, generating a Pigmentation Index of 39.35%!
#
# While the AI correctly predicted the disease (87.1%), the heuristic segmentation 
# failed spectacularly, mislabeling a "Sine Pigmento" (Pigment-less) disease as 
# having 39.35% pigment.
#
# Academic Conclusion:
# This is the final curtain call for pixel-counting. The researcher attempted to 
# patch the grayscale vulnerability by adding `GRAYSCALE_OVERRIDE_LIMIT: 40.0`, 
# but the underlying metric (counting dark pixels) is fundamentally incompatible 
# with multi-modal imaging (Color Fundus vs. FAF vs. Infrared). 
# This proves conclusively that the CDSS must transition to a Modality-Agnostic, 
# Geometric XAI (the Fragmentation Index) that evaluates topology rather than 
# pixel density. The Colab era is officially over.
# ==============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
from datetime import datetime
import hashlib
from google.colab import files

# [Final Colab Complete Implementation Truncated for Archive]
# Full source identical to Final Complete Version codebase.

# TERMINAL OUTPUT ARCHIVE:
# [00:04:04] ⚙️ BOOTING RETINAGUARD V500 (PLATINUM)...
#    ✅ Clinical Engine: ONLINE
#    ✅ Bio-Security Protocol: ONLINE
# 
# [00:04:16] 🚀 PROCESSING: rp_10_sine (3).jpg
# 
# [Visual Dashboard Output: 
#  DIAGNOSIS: POSITIVE (RP)
#  CONFIDENCE: 87.1%
#  FINDING: Moderate Clumping
#  PIGMENT INDEX: 39.35% ]
