"""
================================================================================
DCGAN SIMULATOR - RETINITIS PIGMENTOSA SYNTHETIC IMAGE GENERATOR
================================================================================
This script implements the exact Generator and Discriminator neural network 
architectures described in the Powroźnik et al. (2025) research paper:
"Deep convolutional generative adversarial networks in retinitis pigmentosa 
disease images augmentation and detection"

GENERATOR ARCHITECTURE:
  - Input: 128-dimensional latent noise vector
  - Fully Connected (FC) Layer: 15,360 neurons
  - Reshaping: 6x5x512 tensor
  - 5 cycles of alternating 2D Convolutions and Transposed Convolutions:
    * ConvTrans2D (stride=2, kernel=5x5, Leaky ReLU=0.2) -> 12x10x512
    * Conv2D (stride=1, kernel=5x5, Leaky ReLU=0.2) -> 12x10x256
    * ConvTrans2D (stride=2, kernel=5x5, Leaky ReLU=0.2) -> 24x20x256
    * Conv2D (stride=1, kernel=5x5, Leaky ReLU=0.2) -> 24x20x128
    * ConvTrans2D (stride=2, kernel=5x5, Leaky ReLU=0.2) -> 48x40x128
    * Conv2D (stride=1, kernel=5x5, Leaky ReLU=0.2) -> 48x40x64
    * ConvTrans2D (stride=2, kernel=5x5, Leaky ReLU=0.2) -> 96x80x64
    * Conv2D (stride=1, kernel=5x5, Leaky ReLU=0.2) -> 96x80x32
    * ConvTrans2D (stride=2, kernel=5x5, Leaky ReLU=0.2) -> 192x160x32
    * Conv2D (stride=1, kernel=5x5, Activation=tanh) -> 192x160x1 (Single channel output)

DISCRIMINATOR ARCHITECTURE:
  - Input: 192x160x1 image
  - 5 iterations through Conv2D layers (alternating stride=1 and stride=2)
  - No pooling layers (stride=2 handles downsampling)
  - Activations: Leaky ReLU (slope 0.2)
  - Culminates with two Fully Connected (FC) layers (FC 512 -> FC 1)
  - No final activation (suitable for Wasserstein loss with gradient penalty)
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

class PowroznikGAN:
    """DCGAN implementation as described by Powroźnik et al. (2025)"""
    
    def __init__(self):
        self.latent_dim = 128
        self.image_shape = (192, 160, 1)
        self.generator = None
        self.discriminator = None
        
        if TENSORFLOW_AVAILABLE:
            self.build_generator()
            self.build_discriminator()
            
    def build_generator(self) -> 'Model':
        """Builds the 11-layer Generator with ~15M parameters"""
        model_input = layers.Input(shape=(self.latent_dim,))
        
        # FC Layer: 15,360 neurons
        x = layers.Dense(15360)(model_input)
        x = layers.Reshape((6, 5, 512))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Cycle 1
        x = layers.Conv2DTranspose(512, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(256, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Cycle 2
        x = layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Cycle 3
        x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Cycle 4
        x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Cycle 5 & Output Layer
        x = layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        model_output = layers.Conv2D(1, kernel_size=5, strides=1, padding='same', activation='tanh')(x)
        
        self.generator = Model(model_input, model_output, name="Powroznik_Generator")
        return self.generator

    def build_discriminator(self) -> 'Model':
        """Builds the 11-layer Discriminator with ~9.5M parameters"""
        model_input = layers.Input(shape=self.image_shape)
        
        # Alternating convolution layers with strides 1 and 2
        # Layer 1
        x = layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(model_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 2: Downsample
        x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 3
        x = layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 4: Downsample
        x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 5
        x = layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 6: Downsample
        x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 7
        x = layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 8: Downsample
        x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Layer 9
        x = layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Flatten and Fully Connected Layers
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Output Node (linear activation for Wasserstein loss)
        model_output = layers.Dense(1)(x)
        
        self.discriminator = Model(model_input, model_output, name="Powroznik_Discriminator")
        return self.discriminator

    def generate_synthetic_scan(self, count: int = 1) -> tuple:
        """Generates synthetic Retinitis Pigmentosa structures"""
        if not TENSORFLOW_AVAILABLE:
            # Fallback mock generator using numpy structures
            import numpy as np
            mock_images = np.random.normal(0, 1, (count, 192, 160, 1))
            return mock_images, "MOCK_GAN_OUTPUT"
            
        import numpy as np
        random_latent_vectors = np.random.uniform(0, 1, size=(count, self.latent_dim))
        generated_images = self.generator.predict(random_latent_vectors, verbose=0)
        return generated_images, "TENSORFLOW_GAN_OUTPUT"

if __name__ == "__main__":
    print("================================================================================")
    print("POWROŹNIK ET AL. (2025) DCGAN ARCHITECTURE CHECK")
    print("================================================================================")
    if TENSORFLOW_AVAILABLE:
        gan = PowroznikGAN()
        print(f"[+] TensorFlow Version: {tf.__version__}")
        print("\n--- GENERATOR MODEL SUMMARY ---")
        gan.generator.summary()
        print("\n--- DISCRIMINATOR MODEL SUMMARY ---")
        gan.discriminator.summary()
        
        # Test generation
        images, source = gan.generate_synthetic_scan(1)
        print(f"\n[+] Successfully generated 1 synthetic scan with shape: {images[0].shape} ({source})")
    else:
        print("[!] TensorFlow not available. Initializing rule-based mock model.")
        gan = PowroznikGAN()
        images, source = gan.generate_synthetic_scan(1)
        print(f"[+] Generated mock image structure with shape: {images[0].shape} ({source})")
    print("================================================================================")
