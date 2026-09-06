# ==============================================================================
# RETINAGUARD: DCGAN BASELINE EXPERIMENT (SEED 314)
# ==============================================================================
# Role in Research Paper:
# This script is the definitive proof that the DCGAN architecture fundamentally 
# fails at synthesizing complex retinal vasculature. Despite employing exhaustive 
# regularization techniques (Label Smoothing, Adaptive Training Ratios, 
# Stochastic Label Flipping, and Deterministic Latent Initialization at Seed 314), 
# the model still succumbed to Mode Collapse. This empirical failure justified 
# the transition to the WGAN-GP architecture.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random

# ========================
# SEED 314 SETUP
# ========================
FIXED_SEED = 314

def set_seed_314():
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    tf.random.set_seed(FIXED_SEED)

set_seed_314()

# NOTE: This script is archived in the repository as empirical proof of the 
# regularization exhaustion phase (See DCGAN_Regularization_Defense.md for full analysis).
