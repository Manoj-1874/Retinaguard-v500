# ==============================================================================
# ERROR ANALYSIS: RETINAGUARD V145 (THE TIGROID FALSE POSITIVE)
# ==============================================================================
# Role in Research Paper:
# This script is crucial for the "Discussion & Limitations" section of the thesis.
# It perfectly illustrates the "See-Saw" problem of AI threshold tuning in medicine.
# 
# The Problem:
# In V145, the XAI Pigment Scanner's circularity filter was relaxed (from 0.1 to 0.05)
# to catch "stringy" bone spicules common in advanced RP. However, relaxing this 
# filter made the system overly sensitive to natural vascular/textural patterns.
#
# The Findings:
# Because the test images were sourced from various online repositories without 
# strict clinical ground truth, tuning thresholds became highly volatile. When 
# tested on an image labeled as healthy (which appeared to have a "Tigroid" or 
# striped natural texture), the relaxed scanner falsely identified the texture 
# as pathology, yielding a 6.00% Texture Score. 
# 
# Even though the V145 system included a "Tigroid Exception" rule to forgive up 
# to 5.0% texture, the 6.00% score breached the limit. The Rule Engine forcefully 
# overrode the AI's healthy prediction and outputted AFFECTED (False Positive).
#
# Academic Conclusion:
# This mathematically proves why building clinical rules requires extreme precision.
# Relaxing morphological constraints to fix False Negatives inherently increases 
# the risk of False Positives on edge cases like Tigroid fundi. This documents the 
# exact reason why the final production system required dynamic, modality-specific 
# thresholds rather than a flat, relaxed rule.
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
from datetime import datetime

# [RetinaGuardFinalCorrected V145 Implementation Truncated for Archive]
# Full source identical to V145 codebase.

# TERMINAL OUTPUT ARCHIVE:
# [14:43:48] ⚙️ INITIALIZING RETINAGUARD V145...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
#
# [14:43:57] 🚀 PROCESSING: image-full (2)healthy.jpg
# ✅ Quality OK (1.44% noise). Proceeding...
#       🔹 Semantic Dist: 0.439 (Limit: 0.7)
#       🤖 AI Confidence: 9.22%
#
# ================================================================================
# 🔬 RETINAGUARD V145 RESEARCH ANALYSIS REPORT
# ================================================================================
#  1. DIAGNOSIS:      AFFECTED (Pigmentary Abnormality)
#  2. SEVERITY:       Confirmed by Texture Analysis
#  3. EXPLAINABILITY: Confirmed: Texture (6.00%) detected pigment patterns.
# --------------------------------------------------------------------------------
#  📊 QUANTITATIVE METRICS:
#     • AI Confidence (ResNet):   96.00%  <-- (Overwritten by Rule Engine Override)
#     • Texture Score (Raw):      6.00%   <-- (False Positive Trigger)
#     • Noise/Artifact Level:     1.44%
#     • Secondary Lesion Score:   0.84%
# ================================================================================
