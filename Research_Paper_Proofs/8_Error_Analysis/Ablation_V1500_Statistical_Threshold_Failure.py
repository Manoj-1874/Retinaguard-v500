# ==============================================================================
# ABLATION BENCHMARK: RETINAGUARD V1500 (STATISTICAL THRESHOLD FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents the final attempt to solve the Tigroid False Positive 
# problem using pixel intensity before the team pivoted to the geometric 
# V500.10 Jaggedness solution. 
#
# The Architecture & The Trap:
# Having proven that absolute hard-coded thresholds (like < 40 intensity) fail 
# across different lighting conditions (see V500.19), the team attempted to use 
# "Self-Calibrating Statistical Thresholds". 
# The logic: `adaptive_thresh_val = mean - (1.5 * std_dev)`
# The hypothesis was that the system would dynamically adapt to the lighting of 
# each individual scan.
#
# Findings:
# - SENSITIVITY: 96.21%. A slight improvement over the hard-coded V500.19 (95.5%), 
#   recovering 3 False Negatives.
# - SPECIFICITY: 25.00%. Still catastrophic. It misdiagnosed 6 out of 8 healthy eyes.
#
# Academic Conclusion:
# Why did dynamic statistics fail? Because statistical variance (Standard Deviation) 
# only measures *how much* contrast exists in an image, not *what shape* the contrast 
# takes. A healthy Tigroid eye has a massive standard deviation because the dark 
# vascular stripes contrast sharply with the bright choroid. The adaptive thresholding 
# simply shifted the goalposts, but it still mathematically evaluated smooth stripes 
# and jagged spicules as the exact same thing. 
#
# This is the final, definitive proof that pixel-based thresholding—whether absolute 
# or dynamically statistical—cannot be used for clinical pathology segmentation. 
# Only Geometric Shape Analysis (Aspect Ratio + Solidity, as seen in V500.10) works.
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
import os
import glob
from datetime import datetime
from google.colab import drive
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# [RetinaGuardValidatorV1500 Implementation Truncated for Archive]
# Full source identical to V1500 Adaptive Benchmark codebase.

# TERMINAL OUTPUT ARCHIVE:
# [16:15:12] ⚙️ INITIALIZING RETINAGUARD V1500 (SELF-CALIBRATING)...
#    ✅ RP Classifier Loaded.
# 🚀 STARTING V1500 BENCHMARK (SELF-CALIBRATING)
# Dataset: /content/drive/MyDrive/dataset2
# ============================================================
# 📄 Found 445 images. Scanning...
# [1] ❌ ERROR: image-full (2)healthy.jpg | Truth: 0 -> Pred: 1
# [2] ❌ ERROR: image-full (3)affected.jpg | Truth: 1 -> Pred: 0
# [9] ❌ ERROR: image-full_affected.jpg | Truth: 1 -> Pred: 0
# [13] ❌ ERROR: imagesy(1)affected.jpg | Truth: 1 -> Pred: 0
# [17] ❌ ERROR: 1_right_healthy.jpg | Truth: 0 -> Pred: 1
# [20] ❌ ERROR: hi (1)healthy.jpg | Truth: 0 -> Pred: 1
# [22] ❌ ERROR: download (4)healthy.jpg | Truth: 0 -> Pred: 1
# [23] ❌ ERROR: download (3)healthy.jpg | Truth: 0 -> Pred: 1
# [25] ❌ ERROR: download (1)healthy.jpg | Truth: 0 -> Pred: 1
# [183] ❌ ERROR: rips_rp (1).jpg | Truth: 1 -> Pred: 0
# [200] ❌ ERROR: rips_rp (12).jpg | Truth: 1 -> Pred: 0
# [204] ❌ ERROR: rips_rp (16).jpg | Truth: 1 -> Pred: 0
# [219] ❌ ERROR: rips_rp (3).jpg | Truth: 1 -> Pred: 0
# [220] ❌ ERROR: rips_rp (22).jpg | Truth: 1 -> Pred: 0
# [237] ❌ ERROR: rips_rp (39).jpg | Truth: 1 -> Pred: 0
# [252] ❌ ERROR: rips_rp (57).jpg | Truth: 1 -> Pred: 0
# [266] ❌ ERROR: rips_rp (72).jpg | Truth: 1 -> Pred: 0
# [271] ❌ ERROR: rips_rp (89).jpg | Truth: 1 -> Pred: 0
# [276] ❌ ERROR: rips_rp (76).jpg | Truth: 1 -> Pred: 0
# [279] ❌ ERROR: rips_rp (8).jpg | Truth: 1 -> Pred: 0
# [285] ❌ ERROR: rips_rp (94).jpg | Truth: 1 -> Pred: 0
# [296] ❌ ERROR: rips_rp (97).jpg | Truth: 1 -> Pred: 0
# 
# ============================================================
# 🏆 V1500 FINAL ACCURACY: 94.88%
# 👁️ SENSITIVITY:  96.21%
# 🛡️ SPECIFICITY:  25.00%
# ------------------------------
# TP: 406 | FP: 6 | TN: 2 | FN: 16
# ============================================================
