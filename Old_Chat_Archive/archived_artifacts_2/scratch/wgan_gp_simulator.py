"""
================================================================================
WGAN-GP SIMULATOR - RETINITIS PIGMENTOSA SYNTHETIC IMAGE GENERATOR
================================================================================
This script implements a Wasserstein Generative Adversarial Network with 
Gradient Penalty (WGAN-GP), following the transition from standard GANs 
and DCGANs to solve mode collapse and training instability in medical imaging.

Inspired by:
"WGAN-GP for Synthetic Retinal Image Generation: Enhancing Sensor-Based 
Medical Imaging for Classification Models" (Anaya-Sanchez et al., Sensors 2024)

KEY ADVANTAGES OF WGAN-GP:
  1. Wasserstein Distance Loss: Replaces Jensen-Shannon divergence to provide
     smooth, continuous gradients to the generator, avoiding vanishing gradients.
  2. Gradient Penalty (GP): Enforces the 1-Lipschitz constraint by penalizing
     the norm of the critic gradients relative to a target value of 1.
  3. No Minibatch Discrimination or Sigmoid Activations: The Discriminator
     (renamed Critic) outputs a raw score, not a probability.
================================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class WGAN_GP_Model(Model if TENSORFLOW_AVAILABLE else object):
    """WGAN-GP framework for synthetic retinal fundus images"""
    
    def __init__(self, latent_dim=128, image_shape=(192, 160, 1), gp_weight=10.0):
        super(WGAN_GP_Model, self).__init__() if TENSORFLOW_AVAILABLE else None
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.gp_weight = gp_weight
        self.generator = None
        self.critic = None
        
        if TENSORFLOW_AVAILABLE:
            self.build_generator()
            self.build_critic()
            
    def build_generator(self):
        """Generator upscaling noise vectors to 192x160x1"""
        model_input = layers.Input(shape=(self.latent_dim,))
        
        # Initial projection to 6x5x512
        x = layers.Dense(15360)(model_input)
        x = layers.Reshape((6, 5, 512))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # 5 Upscaling blocks using Conv2DTranspose (Strides=2) and Conv2D (Strides=1)
        # Block 1
        x = layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Block 2
        x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Block 3
        x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Block 4
        x = layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Block 5 & Output mapping
        x = layers.Conv2DTranspose(16, kernel_size=5, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Final Conv mapping to single channel output [-1, 1] using tanh
        model_output = layers.Conv2D(1, kernel_size=5, strides=1, padding='same', activation='tanh')(x)
        
        self.generator = Model(model_input, model_output, name="WGAN_Generator")
        return self.generator
        
    def build_critic(self):
        """Critic network (formerly Discriminator) outputting a scalar score"""
        model_input = layers.Input(shape=self.image_shape)
        
        # Conv layers with Strides=2 for downsampling (No Batch Normalization in WGAN-GP)
        x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same')(model_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Flatten and Dense layers mapping to a single score output (No Sigmoid activation)
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        model_output = layers.Dense(1)(x)
        
        self.critic = Model(model_input, model_output, name="WGAN_Critic")
        return self.critic
        
    def compile_model(self, g_optimizer, c_optimizer):
        """Compiles WGAN-GP model with custom optimizers"""
        if TENSORFLOW_AVAILABLE:
            self.g_optimizer = g_optimizer
            self.c_optimizer = c_optimizer
            
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates WGAN-GP Gradient Penalty to enforce Lipschitz constraint"""
        if not TENSORFLOW_AVAILABLE:
            return 0.0
            
        # Get random interpolation vector alpha
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # Get critic predictions on interpolated images
            critic_predictions = self.critic(interpolated, training=True)
            
        # Calculate gradients of predictions with respect to interpolated images
        gradients = tape.gradient(critic_predictions, [interpolated])[0]
        # Calculate L2 norm of gradients
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        # Compute penalty: deviation from 1
        gp = tf.reduce_mean(tf.square(gradient_l2_norm - 1.0))
        return gp

if __name__ == "__main__":
    print("================================================================================")
    print("ANAYA-SANCHEZ ET AL. (2024) WGAN-GP MODEL VERIFICATION")
    print("================================================================================")
    if TENSORFLOW_AVAILABLE:
        wgan = WGAN_GP_Model()
        print(f"[+] TensorFlow Version: {tf.__version__}")
        print("\n--- CRITIC (DISCRIMINATOR) MODEL SUMMARY ---")
        wgan.critic.summary()
        print("\n--- GENERATOR MODEL SUMMARY ---")
        wgan.generator.summary()
        print("\n[+] Verification Complete: Models built successfully.")
    else:
        print("[!] TensorFlow not available. Initializing mock model parameters.")
        wgan = WGAN_GP_Model()
        print(f"[+] WGAN-GP structure defined. Latent dimension: {wgan.latent_dim}")
    print("================================================================================")
