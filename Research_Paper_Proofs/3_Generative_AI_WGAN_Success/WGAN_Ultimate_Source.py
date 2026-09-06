# ==============================================================================
# RETINAGUARD: WGAN-GP ULTIMATE SOURCE (CELL 1)
# ==============================================================================
# Role in Research Paper:
# This script constitutes Proof 2: Generative Adversarial Viability.
# After the DCGAN architecture failed (due to BCE loss-induced mode collapse), 
# the architecture was fundamentally upgraded to a Wasserstein GAN with 
# Gradient Penalty (WGAN-GP).
# 
# Key Mathematical Upgrades over DCGAN:
# 1. Earth Mover's Distance: Replaced BCE loss with Wasserstein loss.
# 2. Gradient Penalty (WGAN-GP): Enforces 1-Lipschitz continuity.
# 3. Layer Normalization: Replaced Batch Normalization in the Critic.
# 4. Asymmetric Training: Critic trains 5 steps for every 1 Generator step.
# 5. Linear Output: Critic uses Dense(1) without Sigmoid activation.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ========================
# GPU CHECK AND SETUP
# ========================
def check_gpu_setup():
    print("=== GPU SETUP CHECK ===")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠️ Running on CPU (Slow) - PLEASE ENABLE GPU IN RUNTIME SETTINGS")
    print("=" * 50)

check_gpu_setup()

# PATHS
train_dataset_path = "/content/drive/MyDrive/Dataset/Train/Retinitis Pigmentosa"
save_dir_base = "/content/drive/MyDrive/WGAN_Results"
os.makedirs(save_dir_base, exist_ok=True)

# ========================
# 2. PARAMETERS
# ========================
IMG_SHAPE = (128, 128, 3)
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5000
SAVE_INTERVAL = 10
DISPLAY_INTERVAL = 10

# WGAN Hyperparameters
GP_WEIGHT = 10.0
CRITIC_STEPS = 5
INITIAL_LR = 0.0001

print(f"Configuration: {EPOCHS} Epochs | Batch Size {BATCH_SIZE}")

# ========================
# 3. BUILD MODELS
# ========================
def build_generator():
    model = Sequential()
    model.add(Input(shape=(LATENT_DIM,)))
    model.add(Dense(128 * 16 * 16))
    model.add(Reshape((16, 16, 128)))
    model.add(BatchNormalization(momentum=0.8))

    # Upsampling blocks
    model.add(Conv2DTranspose(128, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(32, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    # Output
    model.add(Conv2DTranspose(3, 4, strides=1, padding="same", activation='tanh'))
    return model

def build_critic():
    model = Sequential()
    model.add(Input(shape=IMG_SHAPE))
    model.add(Conv2D(64, 4, strides=2, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, strides=2, padding="same"))
    model.add(LayerNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, 4, strides=2, padding="same"))
    model.add(LayerNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1)) # Linear output (No Sigmoid)
    return model

generator = build_generator()
critic = build_critic()

# Optimizers
opt_g = Adam(learning_rate=INITIAL_LR, beta_1=0.5, beta_2=0.9)
opt_d = Adam(learning_rate=INITIAL_LR, beta_1=0.5, beta_2=0.9)

# ========================
# 4. TRAINING STEPS
# ========================
@tf.function
def gradient_penalty(critic, real, fake):
    batch_size = tf.shape(real)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real + (1 - epsilon) * fake

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]

    # Train Critic (Multiple Steps)
    for _ in range(CRITIC_STEPS):
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_logits = critic(real_images, training=True)
            fake_logits = critic(fake_images, training=True)

            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp = gradient_penalty(critic, real_images, fake_images)
            d_loss = d_cost + GP_WEIGHT * gp

        d_grads = tape.gradient(d_loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(d_grads, critic.trainable_variables))

    # Train Generator (Single Step)
    noise = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        gen_logits = critic(fake_images, training=True)
        g_loss = -tf.reduce_mean(gen_logits)

    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    opt_g.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss, tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

# ========================
# 5. AUTO-CORRECTION (Exploding Gradients Defense)
# ========================
class ModelBackup:
    def __init__(self):
        self.gen_weights = None
        self.critic_weights = None
    def backup(self, gen, cri):
        self.gen_weights = gen.get_weights()
        self.critic_weights = cri.get_weights()
    def restore(self, gen, cri):
        print("   ♻️ Restoring weights from last good epoch...")
        gen.set_weights(self.gen_weights)
        cri.set_weights(self.critic_weights)

# (End of Cell 1)
