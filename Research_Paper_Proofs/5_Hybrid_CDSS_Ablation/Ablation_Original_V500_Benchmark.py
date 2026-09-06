# ==============================================================================
# ABLATION STUDY: ORIGINAL V500 BENCHMARK (WITHOUT GEOMETRIC FILTERS)
# ==============================================================================
# Role in Research Paper:
# To mathematically justify the complex geometric algorithms (Jaggedness Filter, 
# Histogram Lock) introduced in the final V500.10 architecture, we must perform 
# an ablation study. We disable these advanced filters and run the "Original" 
# V500 system on a massive 445-image dataset to observe how the architecture 
# behaves using only naive thresholding and dehazing.
#
# Findings:
# - SENSITIVITY: 99.76%. The base XAI scanner is incredible at finding pigment. 
#   It correctly identified 421 out of 422 diseased patients.
# - SPECIFICITY: 12.50%. The base architecture completely collapsed on healthy 
#   patients. It misdiagnosed 7 out of 8 healthy eyes (including known Tigroid 
#   cases like `1_right_healthy.jpg`) as severe RP.
#
# Academic Conclusion:
# This benchmark proves that high-sensitivity morphological extraction (using 
# CLAHE and adaptive thresholding) is useless on its own because it cannot 
# differentiate between pathological bone spicules and healthy vascular structures. 
# The catastrophic 12.5% specificity justifies the absolute necessity of the 
# V500.10 Jaggedness Check (Aspect Ratio + Solidity) which restored specificity 
# without sacrificing the 99% sensitivity.
# ==============================================================================

import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# [RetinaGuardValidatorV500Original Implementation Truncated for Archive]
# Full source identical to V500 Original Benchmark codebase.

# TERMINAL OUTPUT ARCHIVE:
# [06:01:45] ⚙️ INITIALIZING RETINAGUARD V500 (ORIGINAL)...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
# 🚀 STARTING ORIGINAL V500 BENCHMARK
# Dataset: /content/drive/MyDrive/dataset2
# ============================================================
# 📄 Found 445 images. Scanning...
# [1] ❌ ERROR: image-full (2)healthy.jpg | Truth: 0 -> Pred: 1
# [10] ❌ ERROR: image-square(2)healthy.jpg | Truth: 0 -> Pred: 1
# [17] ❌ ERROR: 1_right_healthy.jpg | Truth: 0 -> Pred: 1
# [20] ❌ ERROR: hi (1)healthy.jpg | Truth: 0 -> Pred: 1
# [22] ❌ ERROR: download (4)healthy.jpg | Truth: 0 -> Pred: 1
# [23] ❌ ERROR: download (3)healthy.jpg | Truth: 0 -> Pred: 1
# [38] ❌ ERROR: cataract_test_affected2.jpg | Truth: 1 -> Pred: 0
# [445] ❌ ERROR: image-squarehealthy.png | Truth: 0 -> Pred: 1
# 
# ============================================================
# 🏆 ORIGINAL V500 ACCURACY: 98.14%
# 👁️ SENSITIVITY:  99.76% (Sick detected as Sick)
# 🛡️ SPECIFICITY:  12.50% (Healthy detected as Healthy)
# ------------------------------
# True Positives: 421 | False Positives: 7
# True Negatives: 1 | False Negatives: 1
# ============================================================
