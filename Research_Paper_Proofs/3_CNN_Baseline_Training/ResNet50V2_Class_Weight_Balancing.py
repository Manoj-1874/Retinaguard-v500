# ==============================================================================
# BASELINE CNN: RESNET50-V2 CLASS WEIGHT BALANCING
# ==============================================================================
# Role in Research Paper:
# This script documents the algorithmic attempt to solve the 0% Specificity 
# (False Positive) problem identified in the previous validation metrics.
#
# The Problem:
# The dataset is severely imbalanced (hundreds of RP images, but very few 
# verified Healthy images). As a result, the standard CNN learned to over-predict 
# RP, causing False Positives on healthy Tigroid retinas.
#
# The Fix (Algorithmic Balancing):
# Before abandoning pure Deep Learning in favor of the Hybrid XAI, this script 
# proves we attempted to fix the CNN mathematically. We used Scikit-Learn's 
# `compute_class_weight` to dynamically penalize the model for missing the 
# rare class (Healthy).
# 
# Calculated Weights:
# - Healthy Weight: 22.00x (Force the model to care 22x more about healthy eyes)
# - RP Weight: 0.51x
#
# Findings:
# By injecting `class_weight=class_weights` into the `.fit()` function, the model 
# achieved a phenomenal 98.44% Validation Accuracy and a perfect 1.0000 AUC. 
# This proves that data imbalance was a core driver of the original 0% Specificity, 
# and mathematical weighting is a valid defense mechanism for clinical CNNs.
# ==============================================================================

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from google.colab import drive

# [ResNet50V2 Balanced Training Implementation Truncated for Archive]
# Full source identical to Class Weight Training codebase.

# TERMINAL OUTPUT ARCHIVE:
# ⚖️ Balancing the Training Scale...
#    Healthy Weight: 22.00x (Pay more attention!)
#    RP Weight:      0.51x
# 
# 🚀 STARTING BALANCED TRAINING...
# Epoch 14/20
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 29s 3s/step - accuracy: 0.9062 - auc: 0.9899 - loss: 0.1250 - val_accuracy: 0.9844 - val_auc: 1.0000 - val_loss: 0.0454
