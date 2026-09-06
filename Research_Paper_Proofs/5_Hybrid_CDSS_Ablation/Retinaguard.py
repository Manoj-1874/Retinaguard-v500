"""
================================================================================
RETINAGUARD V500 - CLINICAL DECISION SUPPORT SYSTEM
================================================================================
Advanced Retinitis Pigmentosa Diagnostic System

ARCHITECTURE:
  - Data Layer: WGAN synthetic data augmentation
  - Model Layer: Deep CNN (VGG16/ResNet/EfficientNet)
  - Logic Layer: 7 Clinical Expert Scanners
  - Decision Layer: Weighted voting + Classic RP Triad verification
  - Output Layer: Text reports + Visual diagnostic panels

CLINICAL COMPONENTS:
  ✅ Classic RP Triad (3 cardinal signs)
  ✅ Supporting evidence (4 additional scanners)
  ✅ Significance multipliers for critical findings
  ✅ Visual proof generation

Author: [Your Name]
Version: 5.0.0 (Final Production)
Date: 2024
================================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tensorflow.keras.models import load_model
from datetime import datetime
from google.colab import files
from google.colab import drive
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
#   GOOGLE DRIVE MOUNT
# ==============================================================================

try:
    drive.mount('/content/drive')
    project_path = '/content/drive/MyDrive/RetinaGuard_Project'
    if not os.path.exists(project_path): 
        os.makedirs(project_path)
    os.chdir(project_path)
    print(f"✅ Current Working Directory: {os.getcwd()}")
except Exception as e:
    print(f"⚠️ Drive Mount Warning: {e}")
    print("⚠️ Running in local mode (reports won't be saved to Drive)")

# ==============================================================================
#   CONFIGURATION - CLINICAL TRIAD SYSTEM
# ==============================================================================

CONFIG = {
    # Model path
    "MODEL_PATH": "/content/drive/MyDrive/RP_Classification_Experiment/Models/RetinaGuard_Clinical_Balanced.h5",
    "INPUT_SIZE": (224, 224),
    
    # EXPERT WEIGHTS - 7 CLINICAL SCANNERS (Total = 1.00)
    "EXPERT_WEIGHTS": {
        # TRIAD COMPONENTS (75% total weight)
        "ai_pattern_recognition": 0.25,     # Overall pattern (includes all triad elements)
        "vessel_attenuation": 0.20,         # TRIAD #2: Arteriolar narrowing
        "pigment_bone_spicules": 0.18,      # TRIAD #1: Bone spicule pigmentation
        "optic_disc_pallor": 0.12,          # TRIAD #3: Waxy disc
        
        # SUPPORTING EVIDENCE (25% total weight)
        "vessel_tortuosity": 0.10,          # Vessel twisting (supportive)
        "texture_degeneration": 0.08,       # Photoreceptor loss
        "spatial_pattern": 0.07,            # Peripheral involvement
    },
    
    # CLINICAL THRESHOLDS
    "CRITICAL_THRESHOLDS": {
        # Vessel Attenuation
        "vessel_severe_attenuation": 0.05,
        "vessel_moderate_attenuation": 0.10,
        "vessel_mild_attenuation": 0.15,
        
        # Pigmentation
        "pigment_extensive": 40,
        "pigment_moderate": 25,
        "pigment_mild": 15,
        
        # Optic Disc Pallor
        "pallor_severe": 210,
        "pallor_moderate": 195,
        "pallor_mild": 180,
        
        # Vessel Tortuosity
        "tortuosity_severe": 1.6,
        "tortuosity_moderate": 1.4,
        
        # Supporting
        "texture_high_entropy": 6.8,
        "spatial_marked_degeneration": 0.60,
        "ai_high_confidence": 0.75,
    },
    
    # SIGNIFICANCE MULTIPLIERS (for critical findings)
    "SIGNIFICANCE_MULTIPLIERS": {
        # Triad findings = highest weight
        "vessel_severe": 2.5,
        "vessel_moderate": 1.6,
        "pigment_extensive": 2.3,
        "pigment_moderate": 1.5,
        "pallor_severe": 2.4,
        "pallor_moderate": 1.6,
        
        # Supporting findings = moderate weight
        "tortuosity_severe": 1.8,
        "tortuosity_moderate": 1.3,
        "texture_irregular": 1.2,
        "spatial_marked": 1.4,
        "ai_high_confidence": 1.5,
    },
    
    # Decision thresholds
    "USE_TTA": True,
    "BASE_THRESHOLD": 0.50,
    "ALERT_THRESHOLD": 0.35,
    "TRIAD_COMPLETE_BONUS": 0.15,  # Bonus if all 3 triad components present
}

# ==============================================================================
#   CLINICAL FINDING CLASS
# ==============================================================================

class ClinicalFinding:
    """Represents a single clinical observation"""
    
    def __init__(self, name, severity, value, normal_range, significance):
        self.name = name
        self.severity = severity  # CRITICAL, MODERATE, MILD, NORMAL
        self.value = value
        self.normal_range = normal_range
        self.significance = significance  # Multiplier for voting weight
    
    def __str__(self):
        icons = {"CRITICAL": "🚨", "MODERATE": "⚠️", "MILD": "⚡", "NORMAL": "✅"}
        icon = icons.get(self.severity, "ℹ️")
        return f"{icon} {self.name}: {self.severity} (value: {self.value:.2f}, normal: {self.normal_range})"

# ==============================================================================
#   EXPERT SYSTEM BASE CLASS
# ==============================================================================

class ExpertSystem:
    """Base class for all clinical expert scanners"""
    
    def __init__(self, name, weight, config):
        self.name = name
        self.base_weight = weight
        self.config = config
        self.reset()
    
    def reset(self):
        """Clear findings from previous diagnosis"""
        self.diagnosis = None
        self.confidence = 0.0
        self.evidence = []
        self.findings = []
        self.significance_multiplier = 1.0
        self.triad_component = False  # Track if this is a triad component
    
    def diagnose(self, img, features):
        """Override in subclass - return vote score"""
        pass
    
    def get_vote(self):
        """Calculate weighted vote with significance multiplier"""
        return self.confidence * self.base_weight * self.significance_multiplier
    
    def add_finding(self, finding):
        """Register a clinical finding"""
        self.findings.append(finding)
        if finding.severity in ["CRITICAL", "MODERATE"]:
            self.significance_multiplier = max(self.significance_multiplier, finding.significance)

# ==============================================================================
#   ADVANCED SCANNERS - TRIAD FOCUSED
# ==============================================================================

class ClinicalTriadScanners:
    """Specialized detectors for RP Triad components"""
    
    @staticmethod
    def extract_optic_disc_pallor(img):
        """
        TRIAD COMPONENT #3: Optic Disc Pallor Detection
        
        Clinical significance:
        - Normal disc: Pink/orange color (L channel: 140-180)
        - Pale disc: Yellowish/white (L channel: 180-200)
        - Waxy disc (RP): Very pale, uniform (L channel: >200)
        
        Algorithm:
        1. Find optic disc (brightest circular region)
        2. Measure lightness in LAB color space
        3. Measure uniformity (low std = waxy)
        4. Measure color (low saturation = pale)
        """
        # Convert to LAB color space (better for brightness)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        h, w = l_channel.shape
        
        # Find optic disc candidate (brightest region in center area)
        center_region = l_channel[h//4:3*h//4, w//4:3*w//4]
        
        # Use morphological top-hat to find bright circular structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(w*0.12), int(h*0.12)))
        tophat = cv2.morphologyEx(center_region, cv2.MORPH_TOPHAT, kernel)
        
        # Find brightest connected component
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate to ensure we capture full disc
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        disc_mask = cv2.dilate(thresh, kernel_small, iterations=2)
        
        if cv2.countNonZero(disc_mask) > 0:
            # Measure disc properties
            disc_pixels_l = center_region[disc_mask > 0]
            
            if len(disc_pixels_l) > 50:  # Ensure we have enough pixels
                disc_brightness = float(np.mean(disc_pixels_l))
                disc_std = float(np.std(disc_pixels_l))
                disc_uniformity = 1.0 / (1.0 + disc_std / 10.0)  # Higher = more uniform
                
                # Also check color saturation in original image
                center_bgr = img[h//4:3*h//4, w//4:3*w//4]
                disc_color = center_bgr[disc_mask > 0]
                
                if len(disc_color) > 0:
                    b_mean = np.mean(disc_color[:, 0])
                    g_mean = np.mean(disc_color[:, 1])
                    r_mean = np.mean(disc_color[:, 2])
                    
                    # Calculate "pinkness" (normal disc has more red/orange)
                    color_saturation = float((r_mean + g_mean * 0.5) / (b_mean + 1))
                    
                    # Pallor score: High brightness + high uniformity + low saturation = waxy
                    pallor_score = disc_brightness * disc_uniformity / (color_saturation + 1)
                else:
                    color_saturation = 1.0
                    pallor_score = disc_brightness
            else:
                disc_brightness = 150.0
                disc_uniformity = 0.5
                color_saturation = 1.0
                pallor_score = 150.0
        else:
            # No disc found - use fallback
            disc_brightness = 150.0
            disc_uniformity = 0.5
            color_saturation = 1.0
            pallor_score = 150.0
        
        # Clinical interpretation
        is_pale = disc_brightness > 195
        is_waxy = (disc_brightness > 210 and disc_uniformity > 0.7)
        
        return {
            'disc_brightness': disc_brightness,
            'disc_uniformity': disc_uniformity,
            'color_saturation': color_saturation,
            'pallor_score': pallor_score,
            'is_pale': is_pale,
            'is_waxy': is_waxy
        }
    
    @staticmethod
    def extract_vessel_tortuosity(vessel_mask):
        """
        SUPPORTING SCANNER: Vessel Tortuosity
        
        Clinical significance:
        - Normal vessels: Relatively straight (tortuosity: 1.0-1.3)
        - Mild twisting: tortuosity 1.3-1.5
        - Moderate twisting (RP): tortuosity 1.5-1.8
        - Severe twisting: tortuosity >1.8
        
        Algorithm:
        1. Skeletonize vessel mask to centerlines
        2. For each vessel segment, calculate arc length / chord length
        3. Higher ratio = more twisted
        """
        if vessel_mask is None or vessel_mask.size == 0:
            return {'mean_tortuosity': 1.0, 'max_tortuosity': 1.0, 'num_tortuous_vessels': 0}
        
        # Skeletonize to get vessel centerlines
        skeleton = skeletonize(vessel_mask // 255).astype(np.uint8) * 255
        
        # Find individual vessel paths
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        tortuosity_scores = []
        tortuous_count = 0
        
        for contour in contours:
            if len(contour) > 30:  # Only analyze vessels longer than 30 pixels
                contour = contour.squeeze()
                
                # Calculate arc length (actual path length)
                arc_length = cv2.arcLength(contour, closed=False)
                
                # Calculate chord length (straight-line distance)
                if len(contour.shape) == 2 and contour.shape[0] > 1:
                    start_point = contour[0]
                    end_point = contour[-1]
                    chord_length = np.linalg.norm(end_point - start_point)
                    
                    if chord_length > 10:  # Avoid tiny vessels
                        tortuosity = arc_length / chord_length
                        tortuosity_scores.append(tortuosity)
                        
                        if tortuosity > 1.5:  # Threshold for "tortuous"
                            tortuous_count += 1
        
        if len(tortuosity_scores) > 0:
            mean_tort = float(np.mean(tortuosity_scores))
            max_tort = float(np.max(tortuosity_scores))
        else:
            mean_tort = 1.0
            max_tort = 1.0
        
        return {
            'mean_tortuosity': mean_tort,
            'max_tortuosity': max_tort,
            'num_tortuous_vessels': tortuous_count,
            'vessel_count_analyzed': len(tortuosity_scores)
        }

# ==============================================================================
#   FEATURE EXTRACTION ENGINE
# ==============================================================================

class FeatureExtractor:
    """Extract all clinically relevant features from fundus images"""
    
    @staticmethod
    def extract_all(img):
        """Extract all features for the expert panel"""
        
        # Core features
        vessel_feats = FeatureExtractor._extract_vessels(img)
        pigment_feats = FeatureExtractor._extract_pigment(img)
        texture_feats = FeatureExtractor._extract_texture(img)
        spatial_feats = FeatureExtractor._extract_spatial(img)
        
        # Triad Component #3
        optic_disc_feats = ClinicalTriadScanners.extract_optic_disc_pallor(img)
        
        # Supporting feature
        tortuosity_feats = ClinicalTriadScanners.extract_vessel_tortuosity(vessel_feats['mask'])
        
        return {
            'vessel': vessel_feats,
            'pigment': pigment_feats,
            'texture': texture_feats,
            'spatial': spatial_feats,
            'optic_disc': optic_disc_feats,
            'vessel_tortuosity': tortuosity_feats
        }
    
    @staticmethod
    def _extract_vessels(img):
        """TRIAD #2: Vessel Attenuation Detection"""
        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(g)
        inverted = cv2.bitwise_not(enhanced)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(opened, cv2.MORPH_TOPHAT, kernel_large)
        
        _, vessel_mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
        
        density = cv2.countNonZero(vessel_mask) / vessel_mask.size
        
        h, w = vessel_mask.shape
        quads = [
            vessel_mask[0:h//2, 0:w//2],
            vessel_mask[0:h//2, w//2:w],
            vessel_mask[h//2:h, 0:w//2],
            vessel_mask[h//2:h, w//2:w]
        ]
        quad_densities = [cv2.countNonZero(q) / q.size for q in quads]
        uniformity = 1 - np.std(quad_densities)
        
        return {'density': density, 'uniformity': uniformity, 'mask': vessel_mask}
    
    @staticmethod
    def _extract_pigment(img):
        """TRIAD #1: Bone Spicule Pigmentation Detection"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        _, dark_mask = cv2.threshold(l_channel, 65, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        
        valid_clusters = 0
        cluster_sizes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < area < 1000:
                valid_clusters += 1
                cluster_sizes.append(area)
        
        return {
            'num_clusters': valid_clusters,
            'cluster_sizes': cluster_sizes,
            'mask': dark_mask
        }
    
    @staticmethod
    def _extract_texture(img):
        """Photoreceptor degeneration texture analysis"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kernel_size = 15
        mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        texture_variance = np.mean(variance)
        
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        return {'variance': texture_variance, 'entropy': entropy}
    
    @staticmethod
    def _extract_spatial(img):
        """Peripheral vs central degradation pattern"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        Y, X = np.ogrid[:h, :w]
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        center_mask = distances < (max_dist * 0.4)
        peripheral_mask = distances > (max_dist * 0.6)
        
        center_brightness = np.mean(gray[center_mask])
        peripheral_brightness = np.mean(gray[peripheral_mask])
        
        if center_brightness > 0:
            peripheral_degradation = (center_brightness - peripheral_brightness) / center_brightness
        else:
            peripheral_degradation = 0.0
        
        return {'peripheral_degradation': peripheral_degradation}

# ==============================================================================
#   EXPERT SYSTEMS - 7 CLINICAL SCANNERS
# ==============================================================================

class AIPatternRecognitionExpert(ExpertSystem):
    """Expert #1: Deep Learning Pattern Recognition"""
    
    def __init__(self, model, weight, config):
        super().__init__("AI Pattern Recognition", weight, config)
        self.model = model
    
    def diagnose(self, img, features):
        # Test-time augmentation
        batch = [
            cv2.resize(img, (224, 224)),
            cv2.resize(cv2.flip(img, 1), (224, 224))
        ]
        batch_arr = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype('float32') / 255.0 for x in batch])
        
        probs = self.model.predict(batch_arr, verbose=0)
        self.confidence = float(np.max(probs))
        self.diagnosis = "RP" if self.confidence > 0.6 else "HEALTHY"
        
        if self.confidence > self.config["CRITICAL_THRESHOLDS"]["ai_high_confidence"]:
            self.add_finding(ClinicalFinding(
                "AI High Confidence RP Detection",
                "CRITICAL",
                self.confidence,
                "<0.75",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["ai_high_confidence"]
            ))
            self.evidence.append(f"🚨 AI STRONGLY suspects RP ({self.confidence*100:.1f}%)")
        elif self.confidence > 0.60:
            self.evidence.append(f"⚠️ AI suggests possible RP ({self.confidence*100:.1f}%)")
        else:
            self.evidence.append(f"Neural network confidence: {self.confidence*100:.1f}%")
        
        return self.get_vote()


class VesselAttenuationExpert(ExpertSystem):
    """Expert #2: TRIAD Component #2 - Vessel Attenuation"""
    
    def __init__(self, weight, config):
        super().__init__("Vessel Attenuation (TRIAD #2)", weight, config)
        self.triad_component = True
    
    def diagnose(self, img, features):
        vessel_feats = features.get('vessel', {})
        density = vessel_feats.get('density', 0.15)
        
        rp_score = 0.0
        
        if density < self.config["CRITICAL_THRESHOLDS"]["vessel_severe_attenuation"]:
            rp_score += 0.50
            self.add_finding(ClinicalFinding(
                "Severe Arteriolar Attenuation",
                "CRITICAL",
                density,
                ">0.05",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["vessel_severe"]
            ))
            self.evidence.append(f"🚨 SEVERE vessel attenuation ({density*100:.1f}%) - TRIAD POSITIVE")
        
        elif density < self.config["CRITICAL_THRESHOLDS"]["vessel_moderate_attenuation"]:
            rp_score += 0.35
            self.add_finding(ClinicalFinding(
                "Moderate Arteriolar Attenuation",
                "MODERATE",
                density,
                ">0.10",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["vessel_moderate"]
            ))
            self.evidence.append(f"⚠️ MODERATE vessel attenuation ({density*100:.1f}%) - TRIAD POSITIVE")
        
        elif density < self.config["CRITICAL_THRESHOLDS"]["vessel_mild_attenuation"]:
            rp_score += 0.15
            self.evidence.append(f"⚡ Mild vessel thinning ({density*100:.1f}%)")
        
        else:
            self.evidence.append(f"✅ Normal vessel density ({density*100:.1f}%) - TRIAD NEGATIVE")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.25 else "HEALTHY"
        return self.get_vote()


class PigmentBoneSpiculesExpert(ExpertSystem):
    """Expert #3: TRIAD Component #1 - Bone Spicule Pigmentation"""
    
    def __init__(self, weight, config):
        super().__init__("Bone Spicule Pigmentation (TRIAD #1)", weight, config)
        self.triad_component = True
    
    def diagnose(self, img, features):
        pigment_feats = features.get('pigment', {})
        num_clusters = pigment_feats.get('num_clusters', 0)
        
        rp_score = 0.0
        
        if num_clusters >= self.config["CRITICAL_THRESHOLDS"]["pigment_extensive"]:
            rp_score += 0.60
            self.add_finding(ClinicalFinding(
                "Extensive Bone Spicule Pigmentation",
                "CRITICAL",
                num_clusters,
                "<40",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["pigment_extensive"]
            ))
            self.evidence.append(f"🚨 EXTENSIVE bone spicules ({num_clusters} clusters) - TRIAD POSITIVE")
        
        elif num_clusters >= self.config["CRITICAL_THRESHOLDS"]["pigment_moderate"]:
            rp_score += 0.40
            self.add_finding(ClinicalFinding(
                "Moderate Bone Spicule Pigmentation",
                "MODERATE",
                num_clusters,
                "<25",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["pigment_moderate"]
            ))
            self.evidence.append(f"⚠️ MODERATE bone spicules ({num_clusters} clusters) - TRIAD POSITIVE")
        
        elif num_clusters >= self.config["CRITICAL_THRESHOLDS"]["pigment_mild"]:
            rp_score += 0.20
            self.evidence.append(f"⚡ Mild pigment changes ({num_clusters} clusters)")
        
        else:
            self.evidence.append(f"✅ Minimal pigmentation ({num_clusters} clusters) - TRIAD NEGATIVE")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.25 else "HEALTHY"
        return self.get_vote()


class OpticDiscPallorExpert(ExpertSystem):
    """Expert #4: TRIAD Component #3 - Optic Disc Pallor"""
    
    def __init__(self, weight, config):
        super().__init__("Optic Disc Pallor (TRIAD #3)", weight, config)
        self.triad_component = True
    
    def diagnose(self, img, features):
        disc_feats = features.get('optic_disc', {})
        brightness = disc_feats.get('disc_brightness', 150)
        uniformity = disc_feats.get('disc_uniformity', 0.5)
        is_waxy = disc_feats.get('is_waxy', False)
        
        rp_score = 0.0
        
        if brightness > self.config["CRITICAL_THRESHOLDS"]["pallor_severe"] and is_waxy:
            rp_score += 0.70
            self.add_finding(ClinicalFinding(
                "Severe Optic Disc Pallor (Waxy)",
                "CRITICAL",
                brightness,
                "<210",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["pallor_severe"]
            ))
            self.evidence.append(f"🚨 SEVERE disc pallor ({brightness:.0f}, waxy) - TRIAD POSITIVE")
        
        elif brightness > self.config["CRITICAL_THRESHOLDS"]["pallor_moderate"]:
            rp_score += 0.45
            self.add_finding(ClinicalFinding(
                "Moderate Optic Disc Pallor",
                "MODERATE",
                brightness,
                "<195",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["pallor_moderate"]
            ))
            self.evidence.append(f"⚠️ MODERATE disc pallor ({brightness:.0f}) - TRIAD POSITIVE")
        
        elif brightness > self.config["CRITICAL_THRESHOLDS"]["pallor_mild"]:
            rp_score += 0.20
            self.evidence.append(f"⚡ Mild disc pallor ({brightness:.0f})")
        
        else:
            self.evidence.append(f"✅ Normal disc appearance ({brightness:.0f}) - TRIAD NEGATIVE")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.25 else "HEALTHY"
        return self.get_vote()


class VesselTortuosityExpert(ExpertSystem):
    """Expert #5: Supporting Evidence - Vessel Tortuosity"""
    
    def __init__(self, weight, config):
        super().__init__("Vessel Tortuosity", weight, config)
    
    def diagnose(self, img, features):
        tort_feats = features.get('vessel_tortuosity', {})
        mean_tort = tort_feats.get('mean_tortuosity', 1.0)
        num_tortuous = tort_feats.get('num_tortuous_vessels', 0)
        
        rp_score = 0.0
        
        if mean_tort > self.config["CRITICAL_THRESHOLDS"]["tortuosity_severe"]:
            rp_score += 0.60
            self.add_finding(ClinicalFinding(
                "Severe Vessel Tortuosity",
                "CRITICAL",
                mean_tort,
                "<1.4",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["tortuosity_severe"]
            ))
            self.evidence.append(f"🚨 SEVERE vessel tortuosity ({mean_tort:.2f})")
        
        elif mean_tort > self.config["CRITICAL_THRESHOLDS"]["tortuosity_moderate"]:
            rp_score += 0.35
            self.add_finding(ClinicalFinding(
                "Moderate Vessel Tortuosity",
                "MODERATE",
                mean_tort,
                "<1.4",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["tortuosity_moderate"]
            ))
            self.evidence.append(f"⚠️ MODERATE vessel tortuosity ({mean_tort:.2f})")
        
        elif mean_tort > 1.3:
            rp_score += 0.15
            self.evidence.append(f"⚡ Mild vessel tortuosity ({mean_tort:.2f})")
        else:
            self.evidence.append(f"✅ Normal vessel curvature ({mean_tort:.2f})")
        
        if num_tortuous > 5:
            rp_score += 0.15
            self.evidence.append(f"   Multiple tortuous vessels detected ({num_tortuous})")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.5 else "HEALTHY"
        return self.get_vote()


class TextureDegenerationExpert(ExpertSystem):
    """Expert #6: Supporting Evidence - Texture Degeneration"""
    
    def __init__(self, weight, config):
        super().__init__("Texture Degeneration", weight, config)
    
    def diagnose(self, img, features):
        texture_feats = features.get('texture', {})
        entropy = texture_feats.get('entropy', 5.5)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / edges.size
        
        rp_score = 0.0
        
        if entropy > self.config["CRITICAL_THRESHOLDS"]["texture_high_entropy"]:
            rp_score += 0.35
            self.add_finding(ClinicalFinding(
                "High Texture Irregularity",
                "MODERATE",
                entropy,
                "<6.8",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["texture_irregular"]
            ))
            self.evidence.append(f"⚠️ High texture irregularity (entropy: {entropy:.2f})")
        elif entropy > 6.3:
            rp_score += 0.20
            self.evidence.append(f"⚡ Moderate texture changes (entropy: {entropy:.2f})")
        else:
            self.evidence.append(f"✅ Smooth texture (entropy: {entropy:.2f})")
        
        if edge_density < 0.03:
            rp_score += 0.25
            self.evidence.append(f"   Low detail density ({edge_density*100:.2f}%) - photoreceptor loss suspected")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.5 else "HEALTHY"
        return self.get_vote()


class SpatialPatternExpert(ExpertSystem):
    """Expert #7: Supporting Evidence - Spatial Pattern"""
    
    def __init__(self, weight, config):
        super().__init__("Spatial Pattern", weight, config)
    
    def diagnose(self, img, features):
        spatial_feats = features.get('spatial', {})
        periph_deg = spatial_feats.get('peripheral_degradation', 0.3)
        
        rp_score = 0.0
        
        if periph_deg > self.config["CRITICAL_THRESHOLDS"]["spatial_marked_degeneration"]:
            rp_score += 0.65
            self.add_finding(ClinicalFinding(
                "Marked Peripheral Degeneration",
                "CRITICAL",
                periph_deg,
                "<0.60",
                self.config["SIGNIFICANCE_MULTIPLIERS"]["spatial_marked"]
            ))
            self.evidence.append(f"🚨 MARKED peripheral degeneration ({periph_deg:.2f})")
        elif periph_deg > 0.50:
            rp_score += 0.40
            self.evidence.append(f"⚠️ MODERATE peripheral changes ({periph_deg:.2f})")
        elif periph_deg > 0.35:
            rp_score += 0.20
            self.evidence.append(f"⚡ Mild peripheral asymmetry ({periph_deg:.2f})")
        else:
            self.evidence.append(f"✅ Symmetric distribution ({periph_deg:.2f})")
        
        self.confidence = min(1.0, rp_score)
        self.diagnosis = "RP" if self.confidence > 0.5 else "HEALTHY"
        return self.get_vote()

# ==============================================================================
#   VISUALIZATION MODULE - SHOW EVIDENCE
# ==============================================================================

class DiagnosticVisualizer:
    """Generate visual proof for diagnosis"""
    
    @staticmethod
    def create_comprehensive_report(img_path, diagnosis_result, features):
        """
        Create a 4-panel diagnostic report:
        1. Original image + diagnosis header
        2. Vessel analysis overlay
        3. Pigmentation heatmap
        4. Optic disc detection
        5. Clinical summary
        """
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # ------------------------------------------------------------------
        # PANEL 1: Original Image + Diagnosis
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(img_rgb)
        ax1.axis('off')
        
        verdict = diagnosis_result['verdict']
        score = diagnosis_result['score']
        confidence = diagnosis_result['confidence']
        
        # Color-coded title
        colors = {
            'CLASSIC_RP': '#d32f2f',
            'RP_POSITIVE': '#f57c00',
            'SUSPICIOUS': '#fbc02d',
            'UNCERTAIN': '#9e9e9e',
            'HEALTHY': '#388e3c'
        }
        title_color = colors.get(verdict, '#000000')
        
        ax1.set_title(
            f"RETINAGUARD V500 DIAGNOSIS\n{verdict.replace('_', ' ')} | Score: {score:.3f} | Confidence: {confidence}",
            fontsize=16,
            fontweight='bold',
            color=title_color,
            pad=20
        )
        
        # ------------------------------------------------------------------
        # PANEL 2: Vessel Analysis
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[1, 0])
        vessel_mask = features['vessel']['mask']
        vessel_density = features['vessel']['density']
        
        # Create colored overlay
        vessel_overlay = img_rgb.copy()
        vessel_overlay[vessel_mask > 0] = [255, 0, 0]  # Red vessels
        blended = cv2.addWeighted(img_rgb, 0.7, vessel_overlay, 0.3, 0)
        
        ax2.imshow(blended)
        ax2.set_title(
            f"VESSEL ANALYSIS (TRIAD #2)\nDensity: {vessel_density*100:.1f}% | Normal: >15%",
            fontsize=12,
            fontweight='bold'
        )
        ax2.axis('off')
        
        # Status indicator
        if vessel_density < 0.05:
            status = "🚨 SEVERE ATTENUATION"
            color = 'red'
        elif vessel_density < 0.10:
            status = "⚠️ MODERATE ATTENUATION"
            color = 'orange'
        elif vessel_density < 0.15:
            status = "⚡ MILD ATTENUATION"
            color = 'yellow'
        else:
            status = "✅ NORMAL"
            color = 'green'
        
        ax2.text(
            0.5, -0.1, status,
            transform=ax2.transAxes,
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=color
        )
        
        # ------------------------------------------------------------------
        # PANEL 3: Pigmentation Detection
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 1])
        pigment_mask = features['pigment']['mask']
        num_clusters = features['pigment']['num_clusters']
        
        # Highlight dark clusters
        pigment_overlay = img_rgb.copy()
        pigment_overlay[pigment_mask > 0] = [255, 255, 0]  # Yellow highlights
        blended = cv2.addWeighted(img_rgb, 0.7, pigment_overlay, 0.3, 0)
        
        ax3.imshow(blended)
        ax3.set_title(
            f"BONE SPICULE PIGMENTATION (TRIAD #1)\nClusters: {num_clusters} | Normal: <15",
            fontsize=12,
            fontweight='bold'
        )
        ax3.axis('off')
        
        # Status indicator
        if num_clusters >= 40:
            status = "🚨 EXTENSIVE"
            color = 'red'
        elif num_clusters >= 25:
            status = "⚠️ MODERATE"
            color = 'orange'
        elif num_clusters >= 15:
            status = "⚡ MILD"
            color = 'yellow'
        else:
            status = "✅ MINIMAL"
            color = 'green'
        
        ax3.text(
            0.5, -0.1, status,
            transform=ax3.transAxes,
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=color
        )
        
        # ------------------------------------------------------------------
        # PANEL 4: Optic Disc Analysis
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[2, 0])
        disc_brightness = features['optic_disc']['disc_brightness']
        is_waxy = features['optic_disc']['is_waxy']
        
        # Convert to LAB and highlight bright regions
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Create heatmap
        heatmap = cv2.applyColorMap(l_channel, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(img_rgb, 0.5, heatmap_rgb, 0.5, 0)
        
        ax4.imshow(blended)
        ax4.set_title(
            f"OPTIC DISC PALLOR (TRIAD #3)\nBrightness: {disc_brightness:.0f} | Normal: 140-180",
            fontsize=12,
            fontweight='bold'
        )
        ax4.axis('off')
        
        # Status indicator
        if disc_brightness > 210 and is_waxy:
            status = "🚨 SEVERE (WAXY)"
            color = 'red'
        elif disc_brightness > 195:
            status = "⚠️ MODERATE PALLOR"
            color = 'orange'
        elif disc_brightness > 180:
            status = "⚡ MILD PALLOR"
            color = 'yellow'
        else:
            status = "✅ NORMAL"
            color = 'green'
        
        ax4.text(
            0.5, -0.1, status,
            transform=ax4.transAxes,
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=color
        )
        
        # ------------------------------------------------------------------
        # PANEL 5: Clinical Summary
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Build text report
        triad_status = diagnosis_result['triad_status']
        findings = diagnosis_result['findings']
        
        summary_text = "CLINICAL SUMMARY\n" + "="*40 + "\n\n"
        
        summary_text += "RP TRIAD STATUS:\n"
        summary_text += f"  {'✅' if triad_status['triad_1'] else '❌'} Bone Spicules: {'POSITIVE' if triad_status['triad_1'] else 'NEGATIVE'}\n"
        summary_text += f"  {'✅' if triad_status['triad_2'] else '❌'} Vessel Attenuation: {'POSITIVE' if triad_status['triad_2'] else 'NEGATIVE'}\n"
        summary_text += f"  {'✅' if triad_status['triad_3'] else '❌'} Optic Disc Pallor: {'POSITIVE' if triad_status['triad_3'] else 'NEGATIVE'}\n\n"
        
        if diagnosis_result['triad_complete']:
            summary_text += "🎯 COMPLETE CLASSIC RP TRIAD\n\n"
        
        summary_text += "KEY FINDINGS:\n"
        if findings:
            for expert_name, finding in findings[:3]:
                summary_text += f"  • {finding.name}\n"
        else:
            summary_text += "  • No critical findings\n"
        
        summary_text += f"\nFINAL SCORE: {score:.3f}\n"
        summary_text += f"CONFIDENCE: {confidence}\n"
        
        ax5.text(
            0.05, 0.95,
            summary_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        plt.tight_layout()
        
        # Save report
        report_filename = f"Report_{os.path.basename(img_path)}"
        plt.savefig(report_filename, dpi=150, bbox_inches='tight')
        print(f"      💾 Saved visual report: {report_filename}")
        
        plt.show()
        
        return report_filename

# ==============================================================================
#   CLINICAL SIGNIFICANCE SYSTEM - MAIN ENGINE
# ==============================================================================

class ClinicalSignificanceSystem:
    """Main diagnostic engine coordinating all expert systems"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.experts = []
        self._initialize()
    
    def _initialize(self):
        """Initialize model and expert panel"""
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 BOOTING RETINAGUARD V500")
        print(f"{'='*70}")
        
        try:
            self.model = load_model(self.config["MODEL_PATH"], compile=False)
            print("   ✅ Deep Learning Model: LOADED")
            print("      (Trained on WGAN-augmented dataset)")
        except Exception as e:
            print(f"   ❌ Model Load Failed: {e}")
            return
        
        weights = self.config["EXPERT_WEIGHTS"]
        
        # Initialize 7 expert systems
        self.experts.append(AIPatternRecognitionExpert(self.model, weights["ai_pattern_recognition"], self.config))
        self.experts.append(VesselAttenuationExpert(weights["vessel_attenuation"], self.config))
        self.experts.append(PigmentBoneSpiculesExpert(weights["pigment_bone_spicules"], self.config))
        self.experts.append(OpticDiscPallorExpert(weights["optic_disc_pallor"], self.config))
        self.experts.append(VesselTortuosityExpert(weights["vessel_tortuosity"], self.config))
        self.experts.append(TextureDegenerationExpert(weights["texture_degeneration"], self.config))
        self.experts.append(SpatialPatternExpert(weights["spatial_pattern"], self.config))
        
        print(f"   ✅ Expert Panel: 7 Clinical Specialists")
        print(f"\n   📋 CLASSIC RP TRIAD COMPONENTS:")
        print(f"      1️⃣  Bone Spicule Pigmentation (18% weight)")
        print(f"      2️⃣  Arteriolar Attenuation (20% weight)")
        print(f"      3️⃣  Optic Disc Pallor (12% weight)")
        print(f"\n   📋 SUPPORTING EVIDENCE:")
        print(f"      • AI Pattern Recognition (25% weight)")
        print(f"      • Vessel Tortuosity (10% weight)")
        print(f"      • Texture Degeneration (8% weight)")
        print(f"      • Spatial Pattern (7% weight)")
        print(f"{'='*70}")
        print(f"   ✅ System Ready!\n")
    
    def diagnose(self, img_path):
        """Perform comprehensive diagnosis on a fundus image"""
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔬 ANALYZING: {os.path.basename(img_path)}")
        print(f"{'='*70}")
        
        img = cv2.imread(img_path)
        if img is None:
            print("   ❌ Failed to load image")
            return None
        
        # Reset all experts
        for expert in self.experts:
            expert.reset()
        
        print(f"\n      🧬 Extracting clinical features...")
        features = FeatureExtractor.extract_all(img)
        
        print(f"\n      👨‍⚕️ EXPERT PANEL CONSULTATION:")
        print(f"      {'-'*66}")
        
        votes = []
        critical_findings = []
        triad_status = {"triad_1": False, "triad_2": False, "triad_3": False}
        
        for expert in self.experts:
            vote = expert.diagnose(img, features)
            votes.append(vote)
            
            # Track triad component status
            if expert.triad_component and expert.confidence > 0.25:
                if "TRIAD #1" in expert.name:
                    triad_status["triad_1"] = True
                elif "TRIAD #2" in expert.name:
                    triad_status["triad_2"] = True
                elif "TRIAD #3" in expert.name:
                    triad_status["triad_3"] = True
            
            for finding in expert.findings:
                if finding.severity in ["CRITICAL", "MODERATE"]:
                    critical_findings.append((expert.name, finding))
            
            status_icon = "🔴" if expert.diagnosis == "RP" else "✅"
            mult_str = f" ×{expert.significance_multiplier:.1f}" if expert.significance_multiplier > 1.0 else ""
            
            print(f"      {status_icon} {expert.name:35s} → {expert.diagnosis:8s} ({expert.confidence*100:5.1f}%{mult_str})")
            print(f"         Vote: {vote:.4f} | {expert.evidence[0] if expert.evidence else 'No findings'}")
        
        base_score = sum(votes)
        triad_complete = all(triad_status.values())
        
        # Apply triad bonus if all 3 components present
        if triad_complete:
            triad_bonus = self.config["TRIAD_COMPLETE_BONUS"]
            base_score += triad_bonus
            print(f"\n      🎯 CLASSIC RP TRIAD COMPLETE! (+{triad_bonus:.3f} bonus)")
        
        has_critical = len([f for e, f in critical_findings if f.severity == "CRITICAL"]) > 0
        
        print(f"\n      🔍 CLINICAL ANALYSIS:")
        print(f"      {'-'*66}")
        print(f"         Base Weighted Score: {sum(votes):.3f}")
        if triad_complete:
            print(f"         Triad Completion Bonus: +{self.config['TRIAD_COMPLETE_BONUS']:.3f}")
        print(f"         Final Composite Score: {base_score:.3f}")
        print(f"\n         RP TRIAD STATUS:")
        print(f"            {'✅' if triad_status['triad_1'] else '❌'} Bone Spicule Pigmentation: {'POSITIVE' if triad_status['triad_1'] else 'NEGATIVE'}")
        print(f"            {'✅' if triad_status['triad_2'] else '❌'} Arteriolar Attenuation: {'POSITIVE' if triad_status['triad_2'] else 'NEGATIVE'}")
        print(f"            {'✅' if triad_status['triad_3'] else '❌'} Optic Disc Pallor: {'POSITIVE' if triad_status['triad_3'] else 'NEGATIVE'}")
        
        if critical_findings:
            print(f"\n         🚨 CRITICAL/MODERATE FINDINGS:")
            for expert_name, finding in critical_findings[:5]:
                print(f"            • {finding}")
        
        print(f"\n      ⚖️  FINAL DIAGNOSIS:")
        print(f"      {'-'*66}")
        
        # Decision logic
        if triad_complete and base_score > self.config["BASE_THRESHOLD"]:
            status = "🔴 POSITIVE - CLASSIC RP (TRIAD COMPLETE)"
            confidence = "VERY HIGH"
            verdict = "CLASSIC_RP"
        elif base_score >= self.config["BASE_THRESHOLD"]:
            status = "🔴 POSITIVE - RP DETECTED"
            confidence = "HIGH" if base_score > 0.70 else "MODERATE"
            verdict = "RP_POSITIVE"
        elif has_critical and base_score > self.config["ALERT_THRESHOLD"]:
            status = "⚠️ SUSPICIOUS - CLINICAL REVIEW REQUIRED"
            confidence = "MODERATE"
            verdict = "SUSPICIOUS"
        elif base_score <= (1 - self.config["BASE_THRESHOLD"]) and not has_critical:
            status = "✅ NEGATIVE - HEALTHY"
            confidence = "HIGH" if base_score < 0.25 else "MODERATE"
            verdict = "HEALTHY"
        else:
            status = "🟡 UNCERTAIN - CLINICAL REVIEW SUGGESTED"
            confidence = "LOW"
            verdict = "UNCERTAIN"
        
        print(f"\n      {status}")
        print(f"      Confidence: {confidence}")
        print(f"      Composite Score: {base_score:.3f}")
        print(f"{'='*70}\n")
        
        return {
            'verdict': verdict,
            'score': base_score,
            'triad_complete': triad_complete,
            'triad_status': triad_status,
            'confidence': confidence,
            'findings': critical_findings,
            'features': features
        }

# ==============================================================================
#   MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Initialize system
    app = ClinicalSignificanceSystem(CONFIG)
    visualizer = DiagnosticVisualizer()
    
    print("\n" + "="*70)
    print("👇 UPLOAD FUNDUS IMAGES FOR CLINICAL TRIAD DIAGNOSIS 👇")
    print("="*70)
    print("\nSYSTEM CAPABILITIES:")
    print("  ✅ WGAN-trained Deep Learning Model")
    print("  ✅ 7-Expert Clinical Panel")
    print("  ✅ Classic RP Triad Verification")
    print("  ✅ Weighted Voting + Significance Multipliers")
    print("  ✅ Visual Diagnostic Reports")
    print("  ✅ Explainable AI Evidence")
    print("="*70 + "\n")
    
    try:
        uploaded = files.upload()
        results = {}
        
        for filename in uploaded.keys():
            # Run diagnosis
            result = app.diagnose(filename)
            
            if result:
                results[filename] = result
                
                # Generate visual report
                print(f"\n      🎨 Generating visual report...")
                report_file = visualizer.create_comprehensive_report(
                    filename, result, result['features']
                )
        
        # Batch summary
        if results:
            print(f"\n{'='*70}")
            print(f"📊 BATCH SUMMARY ({len(results)} images analyzed)")
            print(f"{'='*70}")
            
            for fname, result in results.items():
                triad_str = "✅ TRIAD" if result['triad_complete'] else "❌ partial"
                print(f"{result['verdict']:12s} | {result['score']:.3f} | {triad_str} | {fname}")
            
            print(f"{'='*70}\n")
            print("📥 Visual reports generated and displayed above.")
            print("💾 PNG files saved to current directory.")
    
    except KeyboardInterrupt:
        print("\n❌ Upload cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("✅ RETINAGUARD V500 - READY FOR DEPLOYMENT")
print("="*70)