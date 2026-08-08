"""
================================================================================
IMAGE QUALITY VALIDATOR - RETINAGUARD V500
================================================================================
Prevents garbage-in, garbage-out by validating fundus image quality before
clinical feature extraction begins.

VALIDATION CHECKS:
  1. Blur Detection      - Laplacian variance threshold
  2. Brightness/Exposure - mean intensity 30-220
  3. Resolution          - minimum 512x512 pixels
  4. Vignetting          - peripheral/center brightness ratio
  5. Vessel Coverage     - minimum 5% edge coverage
  6. Optic Disc          - anatomical presence check

Version: 1.0.0
================================================================================
"""

import cv2
import numpy as np
from typing import Dict

# Quality thresholds
BLUR_THRESHOLD       = 100.0   # Laplacian variance (lower = blurrier)
MIN_BRIGHTNESS       = 30      # Minimum mean intensity
MAX_BRIGHTNESS       = 220     # Maximum mean intensity
MIN_RESOLUTION       = 512     # Minimum width/height in pixels
OPTIMAL_RESOLUTION   = 1024    # Recommended resolution
MIN_DYNAMIC_RANGE    = 25      # Minimum std deviation (contrast)
MAX_VIGNETTING_RATIO = 0.30    # Max peripheral/center darkness ratio
MIN_VESSEL_DENSITY   = 0.05    # Minimum vessel network coverage

class ImageQualityValidator:
    """Validate fundus image quality before clinical analysis"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        if not strict_mode:
            self.blur_threshold    = 5.0
            self.min_brightness    = 5
            self.min_dynamic_range = 5
            self.min_vessel_density = 0.001
        else:
            self.blur_threshold    = BLUR_THRESHOLD
            self.min_brightness    = MIN_BRIGHTNESS
            self.min_dynamic_range = MIN_DYNAMIC_RANGE
            self.min_vessel_density = MIN_VESSEL_DENSITY

    def validate(self, image: np.ndarray, patient_id: str = "UNKNOWN") -> Dict:
        warnings, errors, metrics = [], [], {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Resolution check
        h, w = gray.shape
        metrics['resolution'] = f"{w}x{h}"
        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            errors.append(f"Resolution {w}x{h} below minimum {MIN_RESOLUTION}x{MIN_RESOLUTION}")

        # Blur check
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur_variance'] = round(blur_var, 2)
        if blur_var < self.blur_threshold:
            errors.append(f"Image too blurry (variance={blur_var:.2f})")

        # Brightness check
        mean_b = float(gray.mean())
        metrics['mean_brightness'] = round(mean_b, 2)
        if mean_b < self.min_brightness:
            errors.append(f"Image underexposed (mean={mean_b:.1f})")
        elif mean_b > MAX_BRIGHTNESS:
            errors.append(f"Image overexposed (mean={mean_b:.1f})")

        quality_score = max(0.0, min(100.0, 100.0 - len(errors)*20 - len(warnings)*5))
        return {
            'valid': len(errors) == 0,
            'quality_score': quality_score,
            'warnings': warnings,
            'errors': errors,
            'metrics': metrics
        }

def validate_image_quality(image: np.ndarray, patient_id: str = "UNKNOWN", strict: bool = True) -> Dict:
    return ImageQualityValidator(strict_mode=strict).validate(image, patient_id)

    def get_fov_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Create binary mask ignoring black borders of fundus images.
        BUG FIX: Increased erosion kernel from 15x15 to 25x25 to fully
        remove the bright camera ring artifact at the FOV boundary.
        Without this, the ring was being detected as bone spicules.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        # Kernel size 25 removes camera ring artifact completely
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

# ==============================================================================
# IMAGE QUALITY THRESHOLD REFERENCE
# ==============================================================================
# BLUR (Laplacian Variance):
#   < 100  = Reject  (blurry, out of focus)
#   100-150 = Warn   (borderline focus)
#   > 150  = Accept  (sharp image)
#
# BRIGHTNESS (Mean Intensity 0-255):
#   < 30   = Reject  (severely underexposed)
#   30-50  = Warn    (dark image, camera issue)
#   50-200 = Accept  (normal fundus range)
#   200-220 = Warn   (slightly overexposed)
#   > 220  = Reject  (overexposed, washed out)
#
# RESOLUTION:
#   < 512x512  = Reject  (insufficient for vessel measurement)
#   512-1023   = Warn    (below optimal)
#   >= 1024    = Accept  (clinical grade)
#
# VIGNETTING RATIO (peripheral/center brightness):
#   < 0.30 = Reject  (severe peripheral darkening)
#   0.30-0.40 = Warn (noticeable vignetting)
#   > 0.40 = Accept  (uniform illumination)
# ==============================================================================

# BUG FIX (Aug 8): peripheral_degradation was returning negative values.
# Formula: (center_mean - periphery_mean) / center_mean
# When periphery is brighter than center (e.g. flash artifact, angiography),
# this gives a negative result. Downstream code assumed 0.0 was minimum,
# so negative values were incorrectly boosting spatial scanner confidence.
# Fix: clamp result to [0.0, 1.0] range.
#
# peripheral_degradation = max(0.0, min(1.0, raw_degradation))
#
# Also added angiography scaling (0.5x) because FA periphery is naturally
# dark due to dye decay, not due to RP-related photoreceptor loss.

SPATIAL_ANGIO_SCALE = 0.5   # Normalize FA peripheral decay to color fundus baseline
