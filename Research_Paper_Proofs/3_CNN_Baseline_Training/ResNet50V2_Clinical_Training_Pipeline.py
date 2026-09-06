# ==============================================================================
# BASELINE TRAINING: RESNET50-V2 CLINICAL CLASSIFIER
# ==============================================================================
# Role in Research Paper:
# This script documents the exact training pipeline used to generate the 
# foundational Deep Learning model (`RP_Classifier_FYP.h5`) that powers the 
# RetinaGuard hybrid system.
#
# The Architecture:
# - Backbone: Transfer Learning via ResNet50V2 (ImageNet weights frozen).
# - Medical Head: GlobalAveragePooling2D -> Dense(256) -> Dropout(0.5) -> Sigmoid.
# - Clinical Augmentations: Random tilts, zooms, and severe brightness variance 
#   (0.7 - 1.3) to simulate different hospital camera flashes.
#
# The Data Engineering Fix:
# The `flow_from_directory` function was explicitly forced to only read from 
# `['Sorted_Healthy', 'Sorted_RP']`. This was a critical fix because the raw 
# dataset contained noisy/unverified folders (`RP421`, `Review_Needed`) that 
# were corrupting the CNN's latent space during earlier training attempts.
#
# Findings:
# The model achieved phenomenal baseline performance:
# - Training Accuracy: 97.99%
# - Validation Accuracy: 96.88%
# - Validation Sensitivity (Recall): 100.0%
#
# Academic Conclusion:
# This proves that a well-augmented CNN is highly capable of identifying 
# Retinitis Pigmentosa in standard clinical conditions. However, as proven in 
# the Ablation Studies (Proof 4/5), this 96.88% accuracy collapses to near 0% 
# when exposed to Out-of-Distribution images (FAF, UWF, Watermarks), necessitating 
# the deterministic XAI wrapper.
# ==============================================================================

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from google.colab import drive

# [ResNet50V2 Training Pipeline Implementation Truncated for Archive]
# Full source identical to Base CNN Training codebase.

# TERMINAL OUTPUT ARCHIVE:
# ⏳ Loading Datasets (Filtering for Healthy vs RP)...
# Found 352 images belonging to 2 classes.
# Found 87 images belonging to 2 classes.
#    ✅ Class Mapping: {'Sorted_Healthy': 0, 'Sorted_RP': 1}
# 
# 🚀 STARTING CLINICAL TRAINING...
# Epoch 18/20
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 30s 3s/step - accuracy: 0.9799 - auc: 0.9880 - loss: 0.0472 - sensitivity: 0.9902 - val_accuracy: 0.9688 - val_auc: 0.9960 - val_loss: 0.1092 - val_sensitivity: 1.0000
# 🎉 TRAINING COMPLETE!
