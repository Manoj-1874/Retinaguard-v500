"""
================================================================================
RETINAGUARD V500 FLASK API - RETINITIS PIGMENTOSA DIAGNOSTIC SYSTEM
================================================================================
Flask API wrapper for the RetinaGuard V500 Clinical Decision Support System

ARCHITECTURE:
  - 10 Clinical Expert Scanners (expanded from 7)
  - Classic RP Triad Verification (3 cardinal signs)
  - Variant Detection: Sine Pigmento, Punctata Albescens, Sectoral RP
  - Complication Detection: Cystoid Macular Edema (CME)
  - Weighted Voting System
  - Significance Multipliers for Critical Findings
  
Version: 5.2.0 Flask Edition - AI+Consensus Decision Logic
================================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Optional: Load TensorFlow model if available
try:
    import tensorflow as tf
    # Use standalone Keras 3.x to load Keras 3.x models
    import keras
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow not available - using rule-based fallback: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# ==============================================================================
#   CONFIGURATION - CLINICAL TRIAD SYSTEM
# ==============================================================================

CONFIG = {
    # Model path
    "MODEL_PATH": f"{MODEL_PATH}/RetinaGuard_Clinical_Balanced.h5",
    "INPUT_SIZE": (224, 224),

    # EXPERT WEIGHTS - 10 CLINICAL SCANNERS (Total = 1.00)
    "EXPERT_WEIGHTS": {
        # TRIAD COMPONENTS (40% total weight)
        "vessel_attenuation": 0.16,         # TRIAD #2: Arteriolar narrowing
        "pigment_bone_spicules": 0.14,      # TRIAD #1: Bone spicule pigmentation
        "optic_disc_pallor": 0.10,          # TRIAD #3: Waxy disc

        # AI + PATTERN (30% total weight)
        "ai_pattern_recognition": 0.20,     # Overall pattern
        "texture_degeneration": 0.06,       # Photoreceptor loss
        "spatial_pattern": 0.04,            # Peripheral involvement

        # SUPPORTING SCANNERS (15%)
        "vessel_tortuosity": 0.08,          # Vessel twisting
        "quadrant": 0.07,                   # NEW: Sectoral RP detection

        # VARIANT-SPECIFIC SCANNERS (15%)
        "bright_lesion": 0.08,              # NEW: Retinitis Punctata Albescens
        "macula": 0.07,                     # NEW: Cystoid Macular Edema
    },

    # CLINICAL THRESHOLDS
    "CRITICAL_THRESHOLDS": {
        "vessel_severe_attenuation": 0.05,
        "vessel_moderate_attenuation": 0.10,
        "pigment_extensive": 40,
        "pigment_moderate": 25,
        "pallor_severe": 210,
        "pallor_moderate": 195,
        "tortuosity_severe": 1.6,
        "tortuosity_moderate": 1.4,
        "texture_high_entropy": 6.8,
        "spatial_marked_degeneration": 0.60,
        "ai_high_confidence": 0.75,
    },

    # SIGNIFICANCE MULTIPLIERS
    "SIGNIFICANCE_MULTIPLIERS": {
        "vessel_severe": 2.5,
        "vessel_moderate": 1.6,
        "pigment_extensive": 2.3,
        "pigment_moderate": 1.5,
        "pallor_severe": 2.4,
        "pallor_moderate": 1.6,
        "tortuosity_severe": 1.8,
        "tortuosity_moderate": 1.3,
        "texture_irregular": 1.2,
        "spatial_marked": 1.4,
        "ai_high_confidence": 1.5,
    },

    "BASE_THRESHOLD": 0.50,
    "ALERT_THRESHOLD": 0.35,
    "TRIAD_COMPLETE_BONUS": 0.15,
}

# Try to load the model
DEEP_LEARNING_MODEL = None
if TENSORFLOW_AVAILABLE:
    try:
        if os.path.exists(CONFIG["MODEL_PATH"]):
            # Use Keras 3.x directly - model was saved with Keras 3.10
            DEEP_LEARNING_MODEL = keras.models.load_model(CONFIG["MODEL_PATH"], compile=False)
            print(f"✅ Loaded model from {CONFIG['MODEL_PATH']}")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")

# ==============================================================================
#   FEATURE EXTRACTION - Clinical Analysis (FOV-Masked)
# ==============================================================================

def get_fov_mask(img):
    """Creates a mask to ignore the black borders of fundus images.
    BUG FIX #2: Without this, the entire black circular border is detected
    as 'bone spicules' and vessel edges are counted as giant vessels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # Erode slightly to completely remove the bright camera ring artifact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def extract_vessel_features(img, fov_mask):
    """TRIAD #2: Vessel Attenuation Detection
    BUG FIX #3: Apply FOV mask & divide by FOV area, not total image size."""
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(g)
    inverted = cv2.bitwise_not(enhanced)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(opened, cv2.MORPH_TOPHAT, kernel_large)
    
    _, vessel_mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    
    # APPLY FOV MASK to remove border noise
    vessel_mask = cv2.bitwise_and(vessel_mask, fov_mask)
    
    # Correct density: divide by visible retinal area, not total pixels
    fov_area = cv2.countNonZero(fov_mask)
    density = cv2.countNonZero(vessel_mask) / fov_area if fov_area > 0 else 0
    
    return {'density': density, 'mask': vessel_mask}

def extract_pigment_features(img, fov_mask):
    """TRIAD #1: Bone Spicule Pigmentation Detection
    BUG FIX #2: Apply FOV mask so black borders aren't counted as pigment."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Lowered threshold to only catch VERY dark actual pigment deposits
    _, dark_mask = cv2.threshold(l_channel, 45, 255, cv2.THRESH_BINARY_INV)
    
    # APPLY FOV MASK to ignore the black background entirely
    dark_mask = cv2.bitwise_and(dark_mask, fov_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    
    valid_clusters = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 10 < area < 800:  # Filter out tiny noise AND massive shadows
            valid_clusters += 1
    
    return {'num_clusters': valid_clusters, 'mask': dark_mask}

def extract_optic_disc_features(img, fov_mask):
    """TRIAD #3: Optic Disc Pallor Detection
    BUG FIX #4: Find the brightest region ANYWHERE in the FOV instead of
    assuming optic disc is in the center (it's usually off to the side)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_channel = cv2.bitwise_and(l_channel, fov_mask)  # Ignore background
    
    # Look for the brightest 1% of the image (the Optic Disc) anywhere in the FOV
    max_val = np.max(l_channel)
    _, disc_mask = cv2.threshold(l_channel, max_val - 25, 255, cv2.THRESH_BINARY)
    
    # Dilate to capture full disc region
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    disc_mask = cv2.dilate(disc_mask, kernel_small, iterations=2)
    
    if cv2.countNonZero(disc_mask) > 50:
        disc_pixels_l = l_channel[disc_mask > 0]
        disc_brightness = float(np.mean(disc_pixels_l))
        disc_std = float(np.std(disc_pixels_l))
        disc_uniformity = 1.0 / (1.0 + disc_std / 10.0)
        
        # Check color saturation for waxy pallor detection
        img_bgr = cv2.bitwise_and(img, img, mask=disc_mask)
        disc_color = img_bgr[disc_mask > 0]
        b_mean = np.mean(disc_color[:, 0])
        g_mean = np.mean(disc_color[:, 1])
        r_mean = np.mean(disc_color[:, 2])
        color_saturation = float((r_mean + g_mean * 0.5) / (b_mean + 1))
    else:
        disc_brightness = 150.0
        disc_std = 20.0
        disc_uniformity = 0.5
        color_saturation = 1.0
    
    is_pale = disc_brightness > 195
    is_waxy = disc_brightness > 210 and disc_uniformity > 0.7
    
    return {
        'disc_brightness': disc_brightness,
        'disc_uniformity': disc_uniformity,
        'color_saturation': color_saturation,
        'is_pale': is_pale,
        'is_waxy': is_waxy
    }

def extract_texture_features(img, fov_mask):
    """Supporting: Texture Degeneration
    BUG FIX: Apply FOV mask to histogram so black background doesn't skew entropy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Only analyze pixels inside the FOV
    hist, _ = np.histogram(gray[fov_mask > 0], bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {'entropy': entropy}

def extract_spatial_features(img, fov_mask):
    """Supporting: Peripheral vs Central Degradation
    BUG FIX: Now uses FOV mask to exclude background/vignetting artifacts."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Create center and peripheral masks WITHIN the FOV only
    center_mask = (distances < (max_dist * 0.4)) & (fov_mask > 0)
    peripheral_mask = (distances > (max_dist * 0.6)) & (fov_mask > 0)
    
    # Only calculate if we have valid pixels in both regions
    if np.count_nonzero(center_mask) > 100 and np.count_nonzero(peripheral_mask) > 100:
        center_brightness = np.mean(gray[center_mask])
        peripheral_brightness = np.mean(gray[peripheral_mask])
        
        if center_brightness > 0:
            peripheral_degradation = (center_brightness - peripheral_brightness) / center_brightness
        else:
            peripheral_degradation = 0.0
    else:
        peripheral_degradation = 0.0
    
    return {'peripheral_degradation': peripheral_degradation}

def extract_bright_lesion_features(img, fov_mask):
    """NEW SCANNER #1: Retinitis Punctata Albescens Detection
    RPA presents with BRIGHT white/yellowish flecks instead of dark bone spicules.
    This scanner specifically looks for abnormal bright lesions in the retina."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Look for bright pixels (L > 200) that are INSIDE the FOV
    bright_mask = (l_channel > 200) & (fov_mask > 0)
    
    # Also check for yellowish tint (positive b channel = yellow)
    yellow_mask = (b_channel > 135) & (fov_mask > 0)
    
    # Drusen/flecks are small bright spots, not large areas like optic disc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_cleaned = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Count bright lesion clusters
    contours, _ = cv2.findContours(bright_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter: lesions should be small (drusen are 50-300 pixels), not optic disc (>2000px)
    lesion_count = 0
    total_lesion_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 2000:  # Drusen/fleck size range
            lesion_count += 1
            total_lesion_area += area
    
    fov_area = np.sum(fov_mask > 0)
    lesion_density = total_lesion_area / fov_area if fov_area > 0 else 0
    
    # Also check yellow fleck count
    yellow_cleaned = cv2.morphologyEx(yellow_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    yellow_contours, _ = cv2.findContours(yellow_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_fleck_count = sum(1 for cnt in yellow_contours if 20 < cv2.contourArea(cnt) < 2000)
    
    return {
        'lesion_count': lesion_count,
        'lesion_density': lesion_density,
        'yellow_fleck_count': yellow_fleck_count,
        'combined_flecks': lesion_count + yellow_fleck_count
    }

def extract_macula_features(img, fov_mask):
    """NEW SCANNER #2: Cystoid Macular Edema Detection
    CME presents as swelling and fluid cysts in the CENTRAL macula.
    This scanner focuses specifically on the macular region (center 15% of image)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    macula_radius = int(min(h, w) * 0.15)  # Central 15% = macular region
    
    # Create macular mask (circle in center)
    Y, X = np.ogrid[:h, :w]
    macula_mask = np.sqrt((X - center_x)**2 + (Y - center_y)**2) < macula_radius
    macula_mask = macula_mask & (fov_mask > 0)
    
    if np.sum(macula_mask) < 100:
        return {'cme_score': 0, 'macula_irregularity': 0, 'edema_likelihood': 0}
    
    # CME shows irregular texture/cystic spaces in macula
    macula_region = gray[macula_mask]
    macula_std = np.std(macula_region)
    macula_mean = np.mean(macula_region)
    
    # Cysts appear as dark spots within the bright macula
    l_macula = l_channel[macula_mask]
    dark_cyst_ratio = np.sum(l_macula < (np.mean(l_macula) - 20)) / len(l_macula)
    
    # Calculate Local Binary Pattern variance for texture irregularity
    # High variance in the macula = potential CME
    macula_img = gray.copy()
    macula_img[~macula_mask] = 0
    
    # Edge detection in macular region (cysts have internal edges)
    edges = cv2.Canny(macula_img, 30, 100)
    edge_density = np.sum(edges[macula_mask]) / (np.sum(macula_mask) * 255)
    
    # CME score: combines irregularity + dark cyst ratio + edge density
    cme_score = (macula_std / 50) * 0.4 + dark_cyst_ratio * 0.3 + edge_density * 0.3
    
    return {
        'cme_score': min(cme_score, 1.0),
        'macula_irregularity': macula_std,
        'dark_cyst_ratio': dark_cyst_ratio,
        'edge_density': edge_density,
        'edema_likelihood': cme_score
    }

def extract_quadrant_features(img, fov_mask):
    """NEW SCANNER #3: Sectoral RP Detection
    Sectoral RP only affects ONE quadrant (usually inferior/nasal), leaving others normal.
    This scanner analyzes each quadrant independently instead of averaging."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    # Define 4 quadrants: Superior, Inferior, Nasal, Temporal
    # For right eye: Nasal = left side, Temporal = right side
    quadrants = {
        'superior': (slice(0, center_y), slice(0, w)),           # Top half
        'inferior': (slice(center_y, h), slice(0, w)),           # Bottom half
        'nasal': (slice(0, h), slice(0, center_x)),              # Left half (right eye)
        'temporal': (slice(0, h), slice(center_x, w)),           # Right half (right eye)
    }
    
    # Also check diagonal quadrants (superior-nasal, inferior-temporal, etc.)
    diagonal_quadrants = {
        'sup_nasal': (slice(0, center_y), slice(0, center_x)),
        'sup_temporal': (slice(0, center_y), slice(center_x, w)),
        'inf_nasal': (slice(center_y, h), slice(0, center_x)),
        'inf_temporal': (slice(center_y, h), slice(center_x, w)),
    }
    
    quadrant_scores = {}
    
    for name, (y_slice, x_slice) in {**quadrants, **diagonal_quadrants}.items():
        quad_gray = gray[y_slice, x_slice]
        quad_fov = fov_mask[y_slice, x_slice]
        
        # Only analyze inside FOV
        valid_pixels = quad_gray[quad_fov > 0]
        if len(valid_pixels) < 100:
            quadrant_scores[name] = {'brightness': 0, 'degradation': 0}
            continue
        
        brightness = np.mean(valid_pixels)
        std = np.std(valid_pixels)
        
        # Dark regions indicate degeneration
        dark_ratio = np.sum(valid_pixels < 80) / len(valid_pixels)
        
        quadrant_scores[name] = {
            'brightness': brightness,
            'std': std,
            'dark_ratio': dark_ratio,
            'degradation': dark_ratio * (1 - brightness/255)
        }
    
    # Find the WORST quadrant (highest degradation)
    degradations = [q['degradation'] for q in quadrant_scores.values()]
    max_degradation = max(degradations) if degradations else 0
    min_degradation = min(degradations) if degradations else 0
    
    # Sectoral RP: BIG difference between worst and best quadrant
    quadrant_asymmetry = max_degradation - min_degradation
    
    # Find which quadrant is affected
    worst_quadrant = max(quadrant_scores.keys(), key=lambda k: quadrant_scores[k]['degradation'])
    
    # STRICT Sectoral RP Detection:
    # 1. Must have SIGNIFICANT asymmetry (> 0.25, not just 0.15)
    # 2. The worst quadrant must show ACTUAL degradation (> 0.25), not just edge darkness
    # 3. The best quadrant must be relatively healthy (< 0.10)
    is_truly_sectoral = (
        quadrant_asymmetry > 0.25 and  # Much stricter asymmetry threshold
        max_degradation > 0.25 and     # Worst quadrant must be degraded
        min_degradation < 0.10          # Best quadrant must be healthy
    )
    
    return {
        'quadrant_scores': quadrant_scores,
        'max_degradation': max_degradation,
        'min_degradation': min_degradation,
        'quadrant_asymmetry': quadrant_asymmetry,
        'worst_quadrant': worst_quadrant,
        'is_sectoral': is_truly_sectoral  # Much stricter threshold
    }

def preprocess_image(image_data):
    """Decode base64 image and convert to numpy array"""
    try:
        # Handle data URL format
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image.convert('RGB'))
        
        # OpenCV expects BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def detect_angiography(img):
    """Detect if image is fluorescein/ICG angiography instead of color fundus.
    Angiography characteristics:
    - Grayscale or near-grayscale (low color saturation)
    - High contrast (bright vessels on dark background)
    - Black background with bright features
    Returns: (is_angio, confidence, reason)
    """
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Check 1: Color saturation (angiograms are grayscale)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    
    # Check 2: Channel similarity (R≈G≈B in grayscale)
    b, g, r = cv2.split(img)
    rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
    rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))
    max_channel_diff = max(rg_diff, rb_diff, gb_diff)
    
    # Check 3: High contrast (angiograms have very bright and very dark regions)
    std_brightness = np.std(gray)
    
    # Check 4: Inverted histogram (lots of dark pixels, few bright pixels)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[0:80])  # Very dark
    bright_pixels = np.sum(hist[180:256])  # Very bright
    total_pixels = img.shape[0] * img.shape[1]
    dark_ratio = dark_pixels / total_pixels
    
    reasons = []
    score = 0
    
    # Scoring system
    if mean_saturation < 20:  # Very low saturation
        score += 3
        reasons.append(f"Low color saturation ({mean_saturation:.1f})")
    elif mean_saturation < 35:
        score += 1
        reasons.append(f"Reduced saturation ({mean_saturation:.1f})")
    
    if max_channel_diff < 5:  # Channels almost identical (grayscale)
        score += 3
        reasons.append(f"Grayscale image (channel diff: {max_channel_diff:.1f})")
    elif max_channel_diff < 15:
        score += 1
        reasons.append(f"Near-grayscale (channel diff: {max_channel_diff:.1f})")
    
    if std_brightness > 60:  # Very high contrast
        score += 2
        reasons.append(f"High contrast (std: {std_brightness:.1f})")
    
    if dark_ratio > 0.5:  # More than 50% very dark pixels
        score += 2
        reasons.append(f"Predominantly dark ({dark_ratio*100:.1f}% dark pixels)")
    
    # Decision
    is_angio = score >= 5
    confidence = min(score / 10.0, 1.0)
    reason = " | ".join(reasons) if reasons else "Normal color fundus"
    
    return is_angio, confidence, reason

# ==============================================================================
#   EXPERT SYSTEMS - 7 Clinical Scanners
# ==============================================================================

def ai_pattern_recognition_expert(img):
    """Expert #1: AI Pattern Recognition
    BUG FIX #1: Grab RP class probability explicitly, not np.max().
    Uses test-time augmentation (original + horizontal flip) for stability."""
    if DEEP_LEARNING_MODEL is not None:
        # Test-time augmentation: original + horizontal flip
        batch = [
            cv2.resize(img, CONFIG["INPUT_SIZE"]),
            cv2.resize(cv2.flip(img, 1), CONFIG["INPUT_SIZE"]),
        ]
        batch_arr = np.array([
            cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
            for x in batch
        ])
        
        probs = DEEP_LEARNING_MODEL.predict(batch_arr, verbose=0)
        
        # THE PROBABILITY FIX:
        # If output is [Healthy, RP] (2 classes), grab index 1 (RP probability).
        # If it's a single sigmoid output, grab it directly.
        if probs.shape[-1] > 1:
            # Multi-class: index 1 = RP probability, average across TTA
            confidence = float(np.mean(probs[:, 1]))
        else:
            # Single sigmoid: output IS the RP probability
            confidence = float(np.mean(probs))
        
        if confidence > CONFIG["CRITICAL_THRESHOLDS"]["ai_high_confidence"]:
            status = "ANOMALY DETECTED"
            severity = "CRITICAL"
            significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["ai_high_confidence"]
        elif confidence > 0.60:
            status = "SUSPICIOUS"
            severity = "MODERATE"
            significance = 1.3
        elif confidence > 0.30:
            status = "MILD CHANGES"
            severity = "MILD"
            significance = 1.0
        else:
            status = "HEALTHY"
            severity = "NORMAL"
            significance = 1.0
        
        return {
            "status": status,
            "confidence": round(confidence * 100, 1),
            "severity": severity,
            "significance": significance,
            "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["ai_pattern_recognition"] * significance,
            "detail": f"Neural network RP probability: {confidence*100:.1f}%"
        }
    else:
        # Fallback: Rule-based analysis using multiple image features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Analyze overall darkness (RP eyes tend to be darker in periphery)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Edge density (RP has fewer sharp features)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Color channel analysis
        b, g, r = cv2.split(img)
        rg_ratio = np.mean(r.astype(float)) / max(np.mean(g.astype(float)), 1)
        
        # Combine features into confidence score
        # Normal fundus: bright, good edge detail, balanced colors
        score = 0.0
        if mean_brightness < 80:  # Very dark image
            score += 0.25
        elif mean_brightness < 120:
            score += 0.10
        
        if edge_density < 0.03:  # Low detail
            score += 0.15
        
        if std_brightness > 60:  # High contrast variance
            score += 0.10
        
        if rg_ratio > 1.3:  # Reddish tint from pigment
            score += 0.10
        
        confidence = min(score, 0.95)
        
        if confidence > CONFIG["CRITICAL_THRESHOLDS"]["ai_high_confidence"]:
            status = "ANOMALY DETECTED"
            severity = "CRITICAL"
            significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["ai_high_confidence"]
        elif confidence > 0.40:
            status = "SUSPICIOUS"
            severity = "MODERATE"
            significance = 1.3
        elif confidence > 0.20:
            status = "MILD CHANGES"
            severity = "MILD"
            significance = 1.0
        else:
            status = "HEALTHY"
            severity = "NORMAL"
            significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["ai_pattern_recognition"] * significance,
        "detail": f"Rule-based analysis score: {confidence*100:.1f}%"
    }

def vessel_attenuation_expert(features):
    """Expert #2: TRIAD #2 - Vessel Attenuation"""
    density = features['vessel']['density']
    
    if density < CONFIG["CRITICAL_THRESHOLDS"]["vessel_severe_attenuation"]:
        status = "SEVERE ATTENUATION"
        severity = "CRITICAL"
        confidence = 0.95
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["vessel_severe"]
    elif density < CONFIG["CRITICAL_THRESHOLDS"]["vessel_moderate_attenuation"]:
        status = "MODERATE ATTENUATION"
        severity = "MODERATE"
        confidence = 0.80
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["vessel_moderate"]
    elif density < 0.15:
        status = "MILD ATTENUATION"
        severity = "MILD"
        confidence = 0.60
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.20
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["vessel_attenuation"] * significance,
        "detail": f"Vessel density: {density*100:.1f}% (Normal: >15%)",
        "triad_positive": severity in ["CRITICAL", "MODERATE"],
        "triad_component": True
    }

def pigment_bone_spicules_expert(features):
    """Expert #3: TRIAD #1 - Bone Spicule Pigmentation"""
    num_clusters = features['pigment']['num_clusters']
    
    if num_clusters >= CONFIG["CRITICAL_THRESHOLDS"]["pigment_extensive"]:
        status = "EXTENSIVE BONE SPICULES"
        severity = "CRITICAL"
        confidence = 0.95
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["pigment_extensive"]
    elif num_clusters >= CONFIG["CRITICAL_THRESHOLDS"]["pigment_moderate"]:
        status = "MODERATE BONE SPICULES"
        severity = "MODERATE"
        confidence = 0.80
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["pigment_moderate"]
    elif num_clusters >= 15:
        status = "MILD PIGMENTATION"
        severity = "MILD"
        confidence = 0.55
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.15
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["pigment_bone_spicules"] * significance,
        "detail": f"Clusters: {num_clusters} (Normal: <15)",
        "triad_positive": severity in ["CRITICAL", "MODERATE"],
        "triad_component": True
    }

def optic_disc_pallor_expert(features):
    """Expert #4: TRIAD #3 - Optic Disc Pallor"""
    brightness = features['optic_disc']['disc_brightness']
    is_waxy = features['optic_disc']['is_waxy']
    
    if brightness > CONFIG["CRITICAL_THRESHOLDS"]["pallor_severe"] and is_waxy:
        status = "SEVERE PALLOR (WAXY)"
        severity = "CRITICAL"
        confidence = 0.95
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["pallor_severe"]
    elif brightness > CONFIG["CRITICAL_THRESHOLDS"]["pallor_moderate"]:
        status = "MODERATE PALLOR"
        severity = "MODERATE"
        confidence = 0.80
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["pallor_moderate"]
    elif brightness > 180:
        status = "MILD PALLOR"
        severity = "MILD"
        confidence = 0.55
        significance = 1.0
    elif brightness < 140:
        status = "LOW BRIGHTNESS"
        severity = "NORMAL"
        confidence = 0.15
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.20
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["optic_disc_pallor"] * significance,
        "detail": f"Brightness: {brightness:.0f} (Normal: 140-180)",
        "triad_positive": severity in ["CRITICAL", "MODERATE"],
        "triad_component": True
    }

def vessel_tortuosity_expert(features):
    """Expert #5: Supporting - Vessel Tortuosity
    Now uses the actual vessel mask skeleton instead of raw Canny edges."""
    vessel_mask = features.get('vessel', {}).get('mask', None)
    
    if vessel_mask is None or vessel_mask.size == 0:
        mean_tort = 1.0
    else:
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(vessel_mask // 255).astype(np.uint8) * 255
        except ImportError:
            # Fallback: use morphological thinning
            skeleton = cv2.ximgproc.thinning(vessel_mask) if hasattr(cv2, 'ximgproc') else vessel_mask
        
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        tortuosity_scores = []
        for contour in contours:
            if len(contour) > 30:
                contour_array = contour.squeeze()
                arc_length = cv2.arcLength(contour, closed=False)
                if len(contour_array.shape) == 2 and contour_array.shape[0] > 1:
                    start_point = contour_array[0]
                    end_point = contour_array[-1]
                    chord_length = np.linalg.norm(end_point - start_point)
                    if chord_length > 10:
                        tortuosity = arc_length / chord_length
                        tortuosity_scores.append(tortuosity)
        
        mean_tort = float(np.mean(tortuosity_scores)) if len(tortuosity_scores) > 0 else 1.0
    
    if mean_tort > CONFIG["CRITICAL_THRESHOLDS"]["tortuosity_severe"]:
        status = "SEVERE TORTUOSITY"
        severity = "CRITICAL"
        confidence = 0.85
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["tortuosity_severe"]
    elif mean_tort > CONFIG["CRITICAL_THRESHOLDS"]["tortuosity_moderate"]:
        status = "MODERATE TORTUOSITY"
        severity = "MODERATE"
        confidence = 0.70
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["tortuosity_moderate"]
    elif mean_tort > 1.3:
        status = "MILD TORTUOSITY"
        severity = "MILD"
        confidence = 0.50
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.25
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["vessel_tortuosity"] * significance,
        "detail": f"Tortuosity: {mean_tort:.2f} (Normal: <1.3)"
    }

def texture_degeneration_expert(features):
    """Expert #6: Supporting - Texture Degeneration"""
    entropy = features['texture']['entropy']
    
    if entropy > CONFIG["CRITICAL_THRESHOLDS"]["texture_high_entropy"]:
        status = "HIGH IRREGULARITY"
        severity = "MODERATE"
        confidence = 0.70
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["texture_irregular"]
    elif entropy > 6.4:  # Increased from 6.3 to add 0.1 margin
        status = "MODERATE CHANGES"
        severity = "MILD"
        confidence = 0.50
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.25
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["texture_degeneration"] * significance,
        "detail": f"Entropy: {entropy:.2f} (Normal: <6.4)"
    }

def spatial_pattern_expert(features):
    """Expert #7: Supporting - Spatial Pattern"""
    periph_deg = features['spatial']['peripheral_degradation']
    
    if periph_deg > CONFIG["CRITICAL_THRESHOLDS"]["spatial_marked_degeneration"]:
        status = "MARKED PERIPHERAL LOSS"
        severity = "CRITICAL"
        confidence = 0.85
        significance = CONFIG["SIGNIFICANCE_MULTIPLIERS"]["spatial_marked"]
    elif periph_deg > 0.50:
        status = "MODERATE PERIPHERAL LOSS"
        severity = "MODERATE"
        confidence = 0.65
        significance = 1.2
    elif periph_deg > 0.40:  # Increased from 0.35 to add 0.05 margin
        status = "MILD ASYMMETRY"
        severity = "MILD"
        confidence = 0.45
        significance = 1.0
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.20
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["spatial_pattern"] * significance,
        "detail": f"Degradation: {periph_deg:.2f} (Normal: <0.40)"
    }

def bright_lesion_expert(features):
    """Expert #8: Retinitis Punctata Albescens (White Dot Variant)
    Detects the RPA variant that presents with BRIGHT flecks instead of dark pigment."""
    bright = features['bright_lesion']
    combined = bright['combined_flecks']
    density = bright['lesion_density']
    
    # RPA typically shows 50+ flecks scattered across the retina
    if combined > 80 or density > 0.03:
        status = "RPA PATTERN DETECTED"
        severity = "CRITICAL"
        confidence = 0.90
        significance = 2.2  # High significance - this is a clear RP variant
    elif combined > 50 or density > 0.02:
        status = "SIGNIFICANT BRIGHT LESIONS"
        severity = "MODERATE"
        confidence = 0.70
        significance = 1.6
    elif combined > 25 or density > 0.01:
        status = "SCATTERED FLECKS"
        severity = "MILD"
        confidence = 0.45
        significance = 1.2
    else:
        status = "NORMAL"
        severity = "NORMAL"
        confidence = 0.15
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["bright_lesion"] * significance,
        "detail": f"Flecks: {combined} | Density: {density:.4f}"
    }

def macula_expert(features):
    """Expert #9: Cystoid Macular Edema (CME) Detection
    Detects central macular swelling/cysts seen in ~30% of RP patients."""
    macula = features['macula']
    cme_score = macula['cme_score']
    irregularity = macula['macula_irregularity']
    
    if cme_score > 0.60:
        status = "CME SUSPECTED"
        severity = "CRITICAL"
        confidence = 0.85
        significance = 2.0  # CME is a serious complication
    elif cme_score > 0.40:
        status = "MACULAR ABNORMALITY"
        severity = "MODERATE"
        confidence = 0.65
        significance = 1.5
    elif cme_score > 0.25:
        status = "MILD IRREGULARITY"
        severity = "MILD"
        confidence = 0.45
        significance = 1.2
    else:
        status = "NORMAL MACULA"
        severity = "NORMAL"
        confidence = 0.15
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["macula"] * significance,
        "detail": f"CME Score: {cme_score:.2f} | Irregularity: {irregularity:.1f}"
    }

def quadrant_expert(features):
    """Expert #10: Sectoral RP Detection
    Detects RP affecting only one quadrant while others remain normal.
    STRICT THRESHOLDS to avoid false positives from stereo images or natural variation."""
    quad = features['quadrant']
    asymmetry = quad['quadrant_asymmetry']
    worst = quad['worst_quadrant']
    max_deg = quad['max_degradation']
    min_deg = quad['min_degradation']
    is_sectoral = quad['is_sectoral']
    
    # CRITICAL: Only if is_sectoral flag AND very high degradation
    # Requires: asymmetry > 0.25, max_deg > 0.25, min_deg < 0.10
    if is_sectoral and max_deg > 0.35:
        status = f"SECTORAL RP ({worst.upper()})"
        severity = "CRITICAL"
        confidence = 0.85
        significance = 2.0  # Sectoral RP is still RP!
    # MODERATE: Sectoral flag set AND moderate degradation
    elif is_sectoral and max_deg > 0.28:
        status = f"QUADRANT ASYMMETRY ({worst.upper()})"
        severity = "MODERATE"
        confidence = 0.65
        significance = 1.5
    # MILD: Some asymmetry but NOT enough to be clinical
    # Raised threshold from 0.10 to 0.20 to avoid false positives
    elif asymmetry > 0.20 and max_deg > 0.20:
        status = "MILD ASYMMETRY"
        severity = "MILD"
        confidence = 0.35
        significance = 1.0  # No boost for mild
    else:
        status = "SYMMETRIC"
        severity = "NORMAL"
        confidence = 0.10
        significance = 1.0
    
    return {
        "status": status,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "significance": significance,
        "vote": confidence * CONFIG["EXPERT_WEIGHTS"]["quadrant"] * significance,
        "detail": f"Asymmetry: {asymmetry:.2f} | Worst: {worst}"
    }

@app.route('/api/analyze', methods=['POST'])
def analyze_retinal_scan():
    """
    Main endpoint for retinal image analysis
    Expected JSON: { "image": "base64_encoded_image", "patientId": "PT-1234" }
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔬 Analyzing scan for {data.get('patientId', 'Unknown')}")
        print(f"{'='*70}")
        
        # Preprocess image
        img = preprocess_image(data['image'])
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Detect image type (warn if angiography, but continue analysis)
        is_angio, angio_confidence, angio_reason = detect_angiography(img)
        if is_angio:
            print(f"   ⚠️  WARNING: Angiography image detected (confidence: {angio_confidence*100:.1f}%)")
            print(f"   📋 Reason: {angio_reason}")
            print(f"   ℹ️  Continuing analysis with adjusted thresholds...")
        
        # Extract features (with FOV mask to ignore black borders)
        print("   🧬 Extracting clinical features...")
        fov_mask = get_fov_mask(img)
        vessel_feats = extract_vessel_features(img, fov_mask)
        pigment_feats = extract_pigment_features(img, fov_mask)
        optic_disc_feats = extract_optic_disc_features(img, fov_mask)
        texture_feats = extract_texture_features(img, fov_mask)
        spatial_feats = extract_spatial_features(img, fov_mask)
        
        # NEW: Extract features for variant/complication detection
        bright_lesion_feats = extract_bright_lesion_features(img, fov_mask)
        macula_feats = extract_macula_features(img, fov_mask)
        quadrant_feats = extract_quadrant_features(img, fov_mask)
        
        features = {
            'vessel': vessel_feats,
            'pigment': pigment_feats,
            'optic_disc': optic_disc_feats,
            'texture': texture_feats,
            'spatial': spatial_feats,
            'bright_lesion': bright_lesion_feats,
            'macula': macula_feats,
            'quadrant': quadrant_feats
        }
        
        # Run all 10 expert systems
        print("   👨‍⚕️ Expert panel consultation (10 scanners)...")
        print("   " + "-"*66)
        
        ai_result = ai_pattern_recognition_expert(img)
        vessel_result = vessel_attenuation_expert(features)
        pigment_result = pigment_bone_spicules_expert(features)
        optic_result = optic_disc_pallor_expert(features)
        tortuosity_result = vessel_tortuosity_expert(features)
        texture_result = texture_degeneration_expert(features)
        spatial_result = spatial_pattern_expert(features)
        
        # NEW: 3 additional variant/complication scanners
        bright_lesion_result = bright_lesion_expert(features)
        macula_result = macula_expert(features)
        quadrant_result = quadrant_expert(features)
        
        # Print expert results like Colab format
        experts_list = [
            ("AI Pattern Recognition", ai_result),
            ("Vessel Attenuation (TRIAD #2)", vessel_result),
            ("Bone Spicule Pigmentation (TRIAD #1)", pigment_result),
            ("Optic Disc Pallor (TRIAD #3)", optic_result),
            ("Vessel Tortuosity", tortuosity_result),
            ("Texture Degeneration", texture_result),
            ("Spatial Pattern", spatial_result),
            ("Bright Lesions (RPA)", bright_lesion_result),
            ("Macula (CME)", macula_result),
            ("Quadrant (Sectoral)", quadrant_result),
        ]
        for name, r in experts_list:
            icon = "✅" if r['severity'] == 'NORMAL' else "⚠️"
            print(f"   {icon} {name:<40} → {r['status']:<20} ({r['confidence']:>5.1f}%)")
            print(f"      Vote: {r['vote']:.4f} | {r.get('detail', '')}")
        
        # Organize results
        results = {
            "ai_pattern": ai_result,
            "vessels": vessel_result,
            "pigment": pigment_result,
            "optic_disc": optic_result,
            "tortuosity": tortuosity_result,
            "texture": texture_result,
            "spatial": spatial_result,
            "bright_lesion": bright_lesion_result,
            "macula": macula_result,
            "quadrant": quadrant_result
        }
        
        # Check RP Triad Status (FIXED: correct mapping)
        triad_status = {
            "bone_spicules": pigment_result.get('triad_positive', pigment_result['severity'] in ["CRITICAL", "MODERATE"]),
            "vessel_attenuation": vessel_result.get('triad_positive', vessel_result['severity'] in ["CRITICAL", "MODERATE"]),
            "optic_disc_pallor": optic_result.get('triad_positive', optic_result['severity'] in ["CRITICAL", "MODERATE"])
        }
        triad_complete = all(triad_status.values())
        
        # Calculate weighted score (all 10 experts)
        base_score = sum([
            ai_result['vote'],
            vessel_result['vote'],
            pigment_result['vote'],
            optic_result['vote'],
            tortuosity_result['vote'],
            texture_result['vote'],
            spatial_result['vote'],
            bright_lesion_result['vote'],
            macula_result['vote'],
            quadrant_result['vote']
        ])
        
        # ==============================================================================
        # 🧬 VARIANT DETECTION PATHWAYS
        # ==============================================================================
        is_sine_pigmento = False
        is_rpa = False
        is_sectoral = False
        is_cme = False
        
        ai_conf = ai_result['confidence'] / 100.0  # Convert from percentage
        pigment_conf = pigment_result['confidence'] / 100.0
        texture_severity = texture_result['severity']
        bright_severity = bright_lesion_result['severity']
        macula_severity = macula_result['severity']
        quadrant_severity = quadrant_result['severity']
        
        # Pathway #1: Retinitis Punctata Albescens (white flecks instead of dark)
        if bright_severity in ['CRITICAL', 'MODERATE'] and pigment_conf < 0.30:
            is_rpa = True
            rpa_bonus = 0.15
            base_score += rpa_bonus
            print(f"   🔘 RPA PATHWAY ACTIVATED! (+{rpa_bonus:.3f} compensation)")
            print(f"      → Bright lesions detected + No dark bone spicules")
        
        # Pathway #2: Sectoral RP (one quadrant affected)
        # STRICTER: Requires CRITICAL severity AND AI agreement (>50%) to activate
        # This prevents false positives from stereo images or natural variation
        elif quadrant_severity == 'CRITICAL' and ai_conf > 0.50:
            is_sectoral = True
            sectoral_bonus = 0.12
            base_score += sectoral_bonus
            print(f"   📐 SECTORAL RP PATHWAY ACTIVATED! (+{sectoral_bonus:.3f} compensation)")
            print(f"      → Significant quadrant asymmetry: {quadrant_result['detail']}")
            print(f"      → AI agrees: {ai_conf*100:.1f}%")
        
        # Pathway #3: Sine Pigmento (no pigment but AI shows concern + degeneration signs)
        # RELAXED: AI > 0.40 (was 0.65) + texture/spatial showing damage
        # This catches cases where AI isn't confident but sees something
        elif ai_conf > 0.40 and pigment_conf < 0.35 and (
            texture_severity in ['MODERATE', 'CRITICAL'] or 
            spatial_result['severity'] in ['MODERATE', 'CRITICAL']
        ):
            is_sine_pigmento = True
            sine_bonus = 0.18
            base_score += sine_bonus
            print(f"   🧬 SINE PIGMENTO PATHWAY ACTIVATED! (+{sine_bonus:.3f} compensation)")
            print(f"      → AI concerned ({ai_conf*100:.1f}%) + No classic pigment + Degeneration signs")
            print(f"      → Texture: {texture_severity}, Spatial: {spatial_result['severity']}")
        
        # Pathway #4: Classic RP Triad Complete
        elif triad_complete:
            base_score += CONFIG["TRIAD_COMPLETE_BONUS"]
            print(f"   🎯 CLASSIC RP TRIAD COMPLETE! (+{CONFIG['TRIAD_COMPLETE_BONUS']:.3f} bonus)")
        
        # Check for CME complication (can occur with any RP type)
        if macula_severity in ['CRITICAL', 'MODERATE']:
            is_cme = True
            print(f"   💧 CME COMPLICATION DETECTED! ({macula_result['detail']})")
        
        # ==============================================================================
        # 🧠 V503 MEDICAL PRIORITY ENGINE
        # Fixes: AI being muzzled during Sine Pigmento / Early stage cases
        # Strategy: Trust AI more when variant pathways are activated
        # ==============================================================================

        # 1. GATHER INTELLIGENCE
        ai_confidence = ai_result['confidence'] / 100.0  # Convert from percentage
        ai_says_rp = ai_confidence > 0.60  # AI thinks it's RP if >60% confident
        
        # Count how many NON-AI clinical experts flagged this as RP
        # A scanner "votes RP" if it found abnormalities (severity != NORMAL)
        clinical_results = {
            'vessels': vessel_result,
            'pigment': pigment_result,
            'optic_disc': optic_result,
            'tortuosity': tortuosity_result,
            'texture': texture_result,
            'spatial': spatial_result,
            'bright_lesion': bright_lesion_result,
            'macula': macula_result,
            'quadrant': quadrant_result
        }
        # Count MODERATE/CRITICAL as strong votes
        clinical_rp_votes = sum(1 for r in clinical_results.values() if r['severity'] in ['CRITICAL', 'MODERATE'])
        # Also count MILD findings separately
        mild_findings = sum(1 for r in clinical_results.values() if r['severity'] == 'MILD')
        total_clinical_scanners = len(clinical_results)
        
        # Count critical findings
        critical_count = sum(1 for r in clinical_results.values() if r['severity'] == 'CRITICAL')
        
        print(f"   ⚖️  DECISION ENGINE (V503 Medical Priority):")
        print(f"      AI Confidence: {ai_confidence*100:.1f}% | AI Says RP: {'YES' if ai_says_rp else 'NO'}")
        print(f"      Clinical RP Votes: {clinical_rp_votes}/{total_clinical_scanners} | MILD findings: {mild_findings}")
        print(f"      Critical Findings: {critical_count} | Triad Complete: {triad_complete}")
        print(f"      Pathways: SinePigmento={is_sine_pigmento}, RPA={is_rpa}, Sectoral={is_sectoral}")

        # 2. DECISION MATRIX (Medical Priority Order)
        
        # RULE 1: THE GOLD STANDARD (Triad Complete)
        # All 3 Triad components are present. Irrefutable diagnosis.
        if triad_complete:
            verdict = "POSITIVE: CLASSIC RETINITIS PIGMENTOSA (TRIAD COMPLETE)"
            confidence = "VERY HIGH"
            verdict_code = "CLASSIC_RP"
            print(f"      → Rule 1: CLASSIC RP TRIAD")

        # RULE 2: SINE PIGMENTO OVERRIDE (THE FIX FOR PT-1879)
        # If Sine Pigmento pathway activated AND AI shows ANY concern (>40%)
        # RELAXED from 80% to 60% - if pathway flags it, trust it more
        elif is_sine_pigmento and ai_confidence > 0.40:
            verdict = "POSITIVE: RP SINE PIGMENTO (AI CONFIRMED)"
            confidence = "HIGH" if ai_confidence > 0.70 else "MODERATE"
            verdict_code = "RP_SINE_PIGMENTO"
            print(f"      → Rule 2: SINE PIGMENTO OVERRIDE (AI={ai_confidence*100:.1f}% > 40%)")

        # RULE 3: VARIANT PATHWAYS (RPA / Sectoral)
        elif is_rpa:
            verdict = "POSITIVE: RETINITIS PUNCTATA ALBESCENS (RPA)"
            confidence = "HIGH"
            verdict_code = "RP_RPA"
            print(f"      → Rule 3a: RPA VARIANT PATHWAY")
            
        elif is_sectoral:
            verdict = "POSITIVE: SECTORAL RETINITIS PIGMENTOSA"
            confidence = "HIGH"
            verdict_code = "RP_SECTORAL"
            print(f"      → Rule 3b: SECTORAL VARIANT PATHWAY")

        # RULE 4: STRONG AI + PARTIAL CONSENSUS
        # AI agrees (>60%) + at least ONE clinical expert OR critical finding.
        # This balances AI confidence with clinical validation.
        elif ai_says_rp and (clinical_rp_votes >= 1 or critical_count > 0):
            verdict = "POSITIVE: RP DETECTED (VERIFIED)"
            confidence = "HIGH"
            verdict_code = "RP_POSITIVE"
            print(f"      → Rule 4: AI + PARTIAL CONSENSUS (AI=YES, Votes={clinical_rp_votes}, Critical={critical_count})")

        # RULE 5: OVERWHELMING AI (Safety Net)
        # AI is >92% confident. Even if all scanners are silent, trust it.
        # This catches rare variants that fool all physical scanners.
        elif ai_confidence > 0.92:
            verdict = "POSITIVE: RP DETECTED (HIGH AI CONFIDENCE)"
            confidence = "HIGH"
            verdict_code = "RP_POSITIVE"
            print(f"      → Rule 5: OVERWHELMING AI (AI={ai_confidence*100:.1f}% > 92%)")

        # RULE 6: CLINICAL CONSENSUS (No AI Support)
        # 3+ scanners agree, but AI disagrees. Flag as suspicious for review.
        elif clinical_rp_votes >= 3:
            verdict = "SUSPICIOUS: CLINICAL RP SIGNS (AI DISAGREES)"
            confidence = "MODERATE"
            verdict_code = "SUSPICIOUS"
            print(f"      → Rule 6: CLINICAL CONSENSUS WITHOUT AI ({clinical_rp_votes} votes, AI={ai_confidence*100:.1f}%)")

        # RULE 7: CRITICAL PERIPHERAL LOSS (STANDALONE)
        # Spatial shows CRITICAL peripheral loss - this alone warrants investigation
        # Peripheral degeneration is a hallmark of RP and should flag even if AI disagrees
        elif spatial_result['severity'] == 'CRITICAL':
            verdict = "SUSPICIOUS: MARKED PERIPHERAL DEGENERATION"
            confidence = "MODERATE"
            verdict_code = "SUSPICIOUS"
            print(f"      → Rule 7: CRITICAL PERIPHERAL LOSS (Degradation={features['spatial']['peripheral_degradation']:.2f})")

        # RULE 8: PERIPHERAL LOSS + AI CONCERN + OTHER FINDINGS
        # Spatial shows MODERATE + AI shows ANY abnormality (>35%) + at least 1 other finding
        # This catches early/variant RP with peripheral degeneration
        elif spatial_result['severity'] == 'MODERATE' and ai_confidence > 0.35 and (mild_findings >= 1 or clinical_rp_votes >= 1):
            verdict = "SUSPICIOUS: PERIPHERAL DEGENERATION DETECTED"
            confidence = "MODERATE"
            verdict_code = "SUSPICIOUS"
            print(f"      → Rule 8: MODERATE PERIPHERAL LOSS + AI CONCERN (Spatial={spatial_result['severity']}, AI={ai_confidence*100:.1f}%, Mild={mild_findings})")

        # RULE 9: MULTIPLE MILD FINDINGS (NEW)
        # If 3+ scanners show MILD abnormalities + AI shows any concern (>30%)
        # Something is wrong - needs clinical review
        elif mild_findings >= 3 and ai_confidence > 0.30:
            verdict = "SUSPICIOUS: MULTIPLE SUBTLE ANOMALIES"
            confidence = "LOW"
            verdict_code = "SUSPICIOUS"
            print(f"      → Rule 9: MULTIPLE MILD FINDINGS ({mild_findings} mild + AI={ai_confidence*100:.1f}%)")

        # RULE 10: HEALTHY / NEGATIVE
        # None of the above criteria met. Insufficient evidence for RP.
        else:
            verdict = "NEGATIVE: HEALTHY RETINA"
            confidence = "HIGH" if not ai_says_rp and clinical_rp_votes <= 1 and mild_findings <= 2 else "MODERATE"
            verdict_code = "HEALTHY"
            print(f"      → Rule 10: INSUFFICIENT EVIDENCE")

        print(f"   🎯 VERDICT: {verdict_code}")
        print(f"   📊 Score: {base_score:.3f} | Confidence: {confidence}")
        print(f"   📋 Consensus: {clinical_rp_votes}/{total_clinical_scanners} Experts + AI: {'YES' if ai_says_rp else 'NO'}")
        print(f"{'='*70}\n")
        
        # Collect critical findings
        critical_findings = []
        for name, result in results.items():
            if result['severity'] in ['CRITICAL', 'MODERATE']:
                detail = result.get('detail', '')
                critical_findings.append(f"{name}: {result['status']} ({detail})")
        
        # Build expert_opinions array (format frontend expects) - 10 experts
        expert_opinions = [
            {"name": "AI Pattern Recognition", "status": ai_result['status'], "confidence": ai_result['confidence'], "vote": ai_result['vote'], "severity": ai_result['severity'], "detail": ai_result.get('detail', '')},
            {"name": "Vessel Attenuation (TRIAD #2)", "status": vessel_result['status'], "confidence": vessel_result['confidence'], "vote": vessel_result['vote'], "severity": vessel_result['severity'], "detail": vessel_result.get('detail', '')},
            {"name": "Bone Spicule Pigmentation (TRIAD #1)", "status": pigment_result['status'], "confidence": pigment_result['confidence'], "vote": pigment_result['vote'], "severity": pigment_result['severity'], "detail": pigment_result.get('detail', '')},
            {"name": "Optic Disc Pallor (TRIAD #3)", "status": optic_result['status'], "confidence": optic_result['confidence'], "vote": optic_result['vote'], "severity": optic_result['severity'], "detail": optic_result.get('detail', '')},
            {"name": "Vessel Tortuosity", "status": tortuosity_result['status'], "confidence": tortuosity_result['confidence'], "vote": tortuosity_result['vote'], "severity": tortuosity_result['severity'], "detail": tortuosity_result.get('detail', '')},
            {"name": "Texture Degeneration", "status": texture_result['status'], "confidence": texture_result['confidence'], "vote": texture_result['vote'], "severity": texture_result['severity'], "detail": texture_result.get('detail', '')},
            {"name": "Spatial Pattern Analysis", "status": spatial_result['status'], "confidence": spatial_result['confidence'], "vote": spatial_result['vote'], "severity": spatial_result['severity'], "detail": spatial_result.get('detail', '')},
            {"name": "Bright Lesions (RPA)", "status": bright_lesion_result['status'], "confidence": bright_lesion_result['confidence'], "vote": bright_lesion_result['vote'], "severity": bright_lesion_result['severity'], "detail": bright_lesion_result.get('detail', '')},
            {"name": "Macula (CME)", "status": macula_result['status'], "confidence": macula_result['confidence'], "vote": macula_result['vote'], "severity": macula_result['severity'], "detail": macula_result.get('detail', '')},
            {"name": "Quadrant (Sectoral)", "status": quadrant_result['status'], "confidence": quadrant_result['confidence'], "vote": quadrant_result['vote'], "severity": quadrant_result['severity'], "detail": quadrant_result.get('detail', '')}
        ]
        
        # Map status for display - replace non-HEALTHY with standardized terms
        for expert in expert_opinions:
            if expert['severity'] == 'NORMAL':
                expert['status'] = 'HEALTHY'
        
        # Determine overall severity for frontend color coding
        overall_severity = "NORMAL"
        if critical_count > 0 or is_sine_pigmento or is_rpa or is_sectoral:
            overall_severity = "CRITICAL" if critical_count > 0 else "MODERATE"
        elif base_score > CONFIG["ALERT_THRESHOLD"]:
            overall_severity = "MODERATE"
        elif base_score > 0.20:
            overall_severity = "MILD"

        # Add variant findings if detected
        if is_rpa:
            critical_findings.insert(0, "🔘 RPA VARIANT: Retinitis Punctata Albescens detected - white flecks instead of bone spicules")
        if is_sectoral:
            critical_findings.insert(0, f"📐 SECTORAL RP: Disease localized to {quadrant_result.get('detail', 'one quadrant')} - asymmetric degeneration")
        if is_sine_pigmento:
            critical_findings.insert(0, "🧬 SINE PIGMENTO VARIANT: High AI confidence with absent pigmentation - RP without classic bone spicules")
        if is_cme:
            critical_findings.insert(0, "💧 CME COMPLICATION: Cystoid Macular Edema detected - central vision at risk")

        # Prepare response
        response = {
            "patientId": data.get('patientId', 'Unknown'),
            "diagnosis": verdict,
            "severity": overall_severity,
            "expert_opinions": expert_opinions,
            "triad_status": triad_status,
            "triad_complete": triad_complete,
            "is_sine_pigmento": is_sine_pigmento,
            "is_rpa": is_rpa,
            "is_sectoral": is_sectoral,
            "is_cme": is_cme,
            "verdict": verdict,
            "verdict_code": verdict_code,
            "confidence": confidence,
            "composite_score": round(base_score, 3),
            "critical_findings": critical_findings,
            "timestamp": datetime.now().isoformat(),
            "is_angiography": is_angio,
            "angiography_confidence": round(angio_confidence * 100, 1) if is_angio else 0,
            "warning": f"⚠️ ANGIOGRAPHY DETECTED: This appears to be a fluorescein/ICG angiography image. Results may be less reliable than color fundus analysis. ({angio_reason})" if is_angio else None
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "RetinaGuard V500 Flask AI Server - Retinitis Pigmentosa Detection",
        "model_loaded": DEEP_LEARNING_MODEL is not None,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "expert_count": 10,
        "version": "5.3.0"
    }), 200

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Return information about the 10 expert systems"""
    return jsonify({
        "expert_systems": {
            "ai_pattern_recognition": {
                "name": "AI Pattern Recognition",
                "weight": CONFIG["EXPERT_WEIGHTS"]["ai_pattern_recognition"],
                "type": "Deep Learning",
                "loaded": DEEP_LEARNING_MODEL is not None
            },
            "vessel_attenuation": {
                "name": "Vessel Attenuation (TRIAD #2)",
                "weight": CONFIG["EXPERT_WEIGHTS"]["vessel_attenuation"],
                "type": "Triad Component",
                "loaded": True
            },
            "pigment_bone_spicules": {
                "name": "Bone Spicule Pigmentation (TRIAD #1)",
                "weight": CONFIG["EXPERT_WEIGHTS"]["pigment_bone_spicules"],
                "type": "Triad Component",
                "loaded": True
            },
            "optic_disc_pallor": {
                "name": "Optic Disc Pallor (TRIAD #3)",
                "weight": CONFIG["EXPERT_WEIGHTS"]["optic_disc_pallor"],
                "type": "Triad Component",
                "loaded": True
            },
            "vessel_tortuosity": {
                "name": "Vessel Tortuosity",
                "weight": CONFIG["EXPERT_WEIGHTS"]["vessel_tortuosity"],
                "type": "Supporting Evidence",
                "loaded": True
            },
            "texture_degeneration": {
                "name": "Texture Degeneration",
                "weight": CONFIG["EXPERT_WEIGHTS"]["texture_degeneration"],
                "type": "Supporting Evidence",
                "loaded": True
            },
            "spatial_pattern": {
                "name": "Spatial Pattern",
                "weight": CONFIG["EXPERT_WEIGHTS"]["spatial_pattern"],
                "type": "Supporting Evidence",
                "loaded": True
            }
        },
        "triad_system": {
            "enabled": True,
            "complete_bonus": CONFIG["TRIAD_COMPLETE_BONUS"],
            "components": [
                "Bone Spicule Pigmentation",
                "Arteriolar Attenuation",
                "Optic Disc Pallor"
            ]
        }
    }), 200

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING RETINAGUARD V500 FLASK API SERVER")
    print("="*70)
    print("\n📋 CLINICAL DECISION SUPPORT SYSTEM FOR RETINITIS PIGMENTOSA")
    print("\nSYSTEM CAPABILITIES:")
    print("  ✅ 7-Expert Clinical Panel")
    print("  ✅ Classic RP Triad Verification")
    print("  ✅ Weighted Voting System")
    print("  ✅ Significance Multipliers")
    print("  ✅ Real-time Image Analysis")
    print("\nRP TRIAD COMPONENTS:")
    print("  1️⃣  Bone Spicule Pigmentation (18% weight)")
    print("  2️⃣  Arteriolar Attenuation (20% weight)")
    print("  3️⃣  Optic Disc Pallor (12% weight)")
    print("\nSUPPORTING SCANNERS:")
    print("  • AI Pattern Recognition (25% weight)")
    print("  • Vessel Tortuosity (10% weight)")
    print("  • Texture Degeneration (8% weight)")
    print("  • Spatial Pattern (7% weight)")
    print("="*70)
    print(f"\n📡 Server URL: http://localhost:5001")
    print(f"📡 Health Check: http://localhost:5001/api/health")
    print(f"📡 System Info: http://localhost:5001/api/models/info")
    print(f"📡 Analysis Endpoint: POST http://localhost:5001/api/analyze")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
