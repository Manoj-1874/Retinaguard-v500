# ==============================================================================
# FINAL DEPLOYMENT: GOLDEN PACKAGE BUILDER
# ==============================================================================
# Role in Research Paper:
# This script represents the final step of the research notebook. Having proven 
# the Baseline CNN, exposed its flaws via Ablation, and designed the Hybrid 
# RetinaGuard V500 system, this script packages the mathematically optimized 
# CNN (`RetinaGuard_Clinical_Balanced.h5`) for clinical deployment.
#
# The Package:
# It copies the champion model, generates a `model_config.json` file that permanently 
# locks the Youden's J decision threshold (0.6993), and exports the final 
# performance text reports and Confusion Matrix images. Finally, it zips everything 
# into `RetinaGuard_GOLDEN_PACKAGE.zip` for transfer.
#
# Academic Conclusion:
# This finalizes the extraction of the 139MB Google Colab research notebook. 
# The evidentiary chain for the RetinaGuard V500 architecture is now fully secured.
# ==============================================================================

import os
import shutil
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from google.colab import drive, files

# [Golden Package Builder Implementation Truncated for Archive]
# Full source identical to final export codebase.

# TERMINAL OUTPUT ARCHIVE:
# 📦 Creating Final Golden Folder at: /content/drive/MyDrive/RetinaGuard_FINAL_GOLDEN...
#    > Copying the Perfect Model...
#      ✅ Model Saved.
#    > Saving Configuration File...
#      ✅ Config Saved (Threshold Secured).
#    > Generating Final Report Card...
#      ✅ Text Report Saved.
#      ✅ Chart Image Saved.
# 
# 🎁 Zipping files for transfer...
# 👇 DOWNLOADING ZIP FILE NOW...
