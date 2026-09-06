# ==============================================================================
# ERROR ANALYSIS: RETINAGUARD V145 (THE SINE PIGMENTO FAILURE)
# ==============================================================================
# Role in Research Paper:
# This script documents a critical limitation in Pigment-Centric Explainable AI (XAI)
# when dealing with early-stage or rare clinical variants, specifically: 
# Retinitis Pigmentosa Sine Pigmento (RP without pigment).
#
# The Architecture & The Trap:
# The Hybrid CDSS was engineered to find, measure, and map bone spicule pigment.
# However, in "Sine Pigmento" variants, the photoreceptors are dying but the 
# classic black pigment clumps have not yet formed. 
# 
# The system was fed a Sine Pigmento clinical scan:
# 1. The CNN (AI) was highly suspicious, outputting a 53.41% probability. It likely 
#    detected secondary symptoms like vessel attenuation or optic disc pallor.
# 2. The XAI Pigment Scanner measured a Texture Score of 3.34%.
# 
# The Rule Engine Dead Zone:
# V145 had a specific rule for Sine Pigmento:
# `elif final_ai_score > 0.50 and rp_score < 1.0: diagnosis = "SUSPICIOUS"`
# However, because the texture score was 3.34% (likely picking up natural veins), 
# it bypassed the Sine Pigmento rule (which required <1.0%), but it also bypassed 
# the AFFECTED rule (which required >4.0%).
#
# Falling through the logic gaps, the system defaulted to "HEALTHY", resulting in 
# a False Negative on a diseased patient.
#
# Academic Conclusion:
# This mathematically proves the danger of rigid, hard-coded clinical thresholds 
# ("Dead Zones"). It also highlights the fundamental limitation of morphological XAI: 
# If a disease variant presents without its hallmark visual symptom (pigment), a 
# feature-extraction algorithm will fail, requiring a much tighter integration with 
# the CNN's latent space to catch "invisible" or occult disease markers.
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

# [RetinaGuardFinalCorrected V145 (Staging Patched) Truncated for Archive]
# Full source identical to V145 Staging Patched codebase.

# TERMINAL OUTPUT ARCHIVE:
# [11:46:09] ⚙️ INITIALIZING RETINAGUARD V145 (STAGING PATCHED)...
#    ✅ RP Classifier Loaded.
#    ✅ ResNet50 Guard Loaded.
#
# [11:46:16] 🚀 PROCESSING: Retinitis-pigmentosa-sine-pigmento-OS_Affected.jpg
# ✅ Quality OK (1.54% noise, Sharpness: 91.27).
#       🔹 Semantic Dist: 0.530 (Limit: 0.7)
#       🤖 AI Confidence: 53.41%
#
# ================================================================================
# 🔬 RETINAGUARD V145 RESEARCH ANALYSIS REPORT
# ================================================================================
#  1. DIAGNOSIS:      HEALTHY
#  2. SEVERITY:       None
#  3. CLINICAL STAGE: N/A (Healthy)
#  4. EXPLAINABILITY: No significant pathology detected.
# --------------------------------------------------------------------------------
#  📊 QUANTITATIVE METRICS:
#     • AI Confidence (ResNet):   53.41%  <-- (AI was suspicious!)
#     • Texture Score (Raw):      3.34%   <-- (Fell into the 1.0% to 4.0% Dead Zone)
#     • Noise/Artifact Level:     1.54%
#     • Secondary Lesion Score:   0.14%
# ================================================================================
