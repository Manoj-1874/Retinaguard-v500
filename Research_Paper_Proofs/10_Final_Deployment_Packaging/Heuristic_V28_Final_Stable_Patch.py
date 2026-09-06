# ==============================================================================
# FINAL DEPLOYMENT PACKAGING: HEURISTIC V28 FINAL STABLE PATCH
# ==============================================================================
# Role in Research Paper:
# This script documents the final, desperate heuristic patch to fix the catastrophic 
# failure of V27. After realizing that circularity metrics were too fragile for 
# wide-field mosaics (peanuts), the researcher completely abandoned them.
#
# The Experiment:
# In V28, the researcher replaced the complex contour geometry with a simple 
# Aspect Ratio check (`aspect_ratio > 1.4`) to detect Mosaics. To detect Grids, 
# they implemented a literal "Black Cross" detector: checking if the exact center 
# pixel patch of the image was dark (`< 40`).
#
# Findings:
# When fed `norm_3.jpg` (the mosaic that crashed V27), the Aspect Ratio check 
# successfully triggered (Ratio = 1.56). The sliding window properly cropped the image 
# without distortion. The AI correctly scored it as 10.0% (Negative/Healthy).
# 
# However, the script reveals yet another hilarious heuristic logic bug: 
# The CDSS recommendation engine uses `if confidence < 0.80` to flag uncertain scans. 
# Because the AI was 10.0% confident it was sick (meaning it was 90.0% confident it 
# was healthy), the CDSS blindly triggered the "Low Confidence" warning, advising 
# manual fundoscopy for a perfectly healthy, highly-confident scan.
#
# Academic Conclusion:
# This script is the perfect bookend to the Colab era. It proves that heuristic 
# programming is an endless treadmill. Fixing the Mosaic distortion (by switching 
# from Circularity to Aspect Ratio) just introduced a new, absurdly brittle Grid 
# detector (the "Black Cross") and exposed a logic flaw in the UI recommendation engine. 
# You cannot build a robust medical device on a foundation of duct tape.
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
from datetime import datetime
import hashlib

# [V28 Final Stable Patch Implementation Truncated for Archive]
# Full source identical to Final V28 Stable codebase.

# TERMINAL OUTPUT ARCHIVE:
# [10:43:41] ⚙️ BOOTING RETINAGUARD V500 (CDSS ONLINE)...
#    ✅ Clinical Engine: ONLINE
# 
# 👇 UPLOAD FILES FOR FINAL MEDICAL REPORT WITH RECOMMENDATIONS 👇
# [File Uploaded: norm_3 (2).jpg]
# [10:43:50] 🚀 PROCESSING: norm_3 (2).jpg
#       🩺 DIAGNOSTICS: Ratio=1.56 | Center Brightness=89.1
#       🥜 MOSAIC DETECTED: Initiating Sliding Window...
#
# [MATPLOTLIB UI RENDERED: NEGATIVE (HEALTHY) - 10.0% Confidence - Pigment: 0.90%]
# [RECOMMENDATION: 🔍 LOW CONFIDENCE: Image quality or artifact may affect AI. Manual Fundoscopy Required.]
