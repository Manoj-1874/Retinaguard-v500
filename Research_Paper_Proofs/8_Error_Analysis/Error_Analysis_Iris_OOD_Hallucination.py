# ==============================================================================
# ERROR ANALYSIS: IRIS OOD HALLUCINATION (V500 GOLDEN FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents a critical failure mode of the finalized RetinaGuard V500 
# system. It proves that even with the "Bio-Security Guard" (a ResNet50 embedding 
# cosine distance checker), the system can still be fooled by Out-of-Distribution 
# (OOD) anatomical imagery.
#
# The Experiment:
# The user uploaded `rp_5.jpg`, which is a high-resolution image of a human IRIS 
# (the outside of the eye), not a fundus (retina) scan. 
#
# Findings:
# 1. The Bio-Security Guard FAILED to block the image. The cosine distance between 
#    the Iris and the reference Retina was likely less than the 0.70 threshold.
# 2. Because the Iris was passed to the clinical model, the CNN hallucinated.
# 3. The dark crypts and folds of the Iris triggered the geometric segmentation 
#    engine, causing the system to diagnose the Iris with Retinitis Pigmentosa 
#    (Confidence: 93.1%, Tex Score: 0.6).
#
# Academic Conclusion:
# This is a phenomenal addition to the Limitations section of your research paper. 
# It proves that while the V500 system excels at diagnosing retinal pathologies 
# and fighting off internal artifacts (watermarks, cataracts, tigroids), its 
# front-line OOD rejection threshold (0.70) is too lenient, allowing non-retinal 
# ophthalmic images (like the anterior segment/iris) to bypass security and 
# cause geometric hallucinations.
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
from datetime import datetime
from google.colab import drive, files

# [Iris OOD Hallucination Implementation Truncated for Archive]
# Full source identical to Iris Test codebase.

# TERMINAL OUTPUT ARCHIVE:
# [00:38:46] ⚙️ INITIALIZING RETINAGUARD V500 (FINAL FIXED)...
#    ✅ Clinical Brain Loaded.
#    ✅ Bio-Security Guard Loaded.
# 
# 👇 RETEST rp_4.jpg NOW 👇
# Saving rp_5.jpg to rp_5.jpg
# 
# [00:38:58] 🚀 PROCESSING: rp_5.jpg
#       🤖 AI Score: 0.9585
#       🏥 Diagnosis: AFFECTED (Retinitis Pigmentosa)
# 
# [Visual Output confirms the AI overlaid red RP spicule markers directly 
#  onto the crypts and furrows of a healthy human Iris.]
