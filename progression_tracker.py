"""
================================================================================
PROGRESSION TRACKER - RETINAGUARD V500
================================================================================
Compares serial fundus scans to detect RP progression over time.
RP is a progressive disease - demonstrating worsening is key to true diagnosis.

METHOD:
  1. Image registration via ORB feature matching + RANSAC homography
  2. Vessel density delta    (typical RP: 5-10% loss per year)
  3. Pigment cluster delta   (accumulation over time)
  4. Spatial degradation delta

PROGRESSION CATEGORIES:
  RAPID    >= 15%/year  -> Urgent referral
  MODERATE  8-15%/year  -> Standard monitoring
  SLOW      3-8%/year   -> Typical RP
  STABLE  < 3%/year     -> Possible treatment success

MINIMUM INTERVAL: 6 months (180 days) between scans for reliable analysis.

Version: 1.0.0
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

RAPID_THRESHOLD    = 0.15
MODERATE_THRESHOLD = 0.08
SLOW_THRESHOLD     = 0.03
MIN_DAYS_BETWEEN   = 180    # 6 months minimum

class ProgressionTracker:

    def compare(self, baseline: np.ndarray, current: np.ndarray,
                months_between: int = 12,
                baseline_features: Optional[Dict] = None,
                current_features:  Optional[Dict] = None) -> Dict:

        if months_between * 30 < MIN_DAYS_BETWEEN:
            return {
                "error": f"Minimum {MIN_DAYS_BETWEEN} days required between scans. Got {months_between*30} days.",
                "progression_category": "INSUFFICIENT_INTERVAL",
            }

        registered, success, confidence = self._register_images(baseline, current)
        vessel_change   = self._compare_vessel_density(baseline_features, current_features)
        pigment_change  = self._compare_pigment_clusters(baseline_features, current_features)
        spatial_change  = self._compare_spatial_degradation(baseline_features, current_features)

        annual_rate = abs(vessel_change) * (12 / max(months_between, 1))

        if annual_rate >= RAPID_THRESHOLD:
            category = "RAPID"
            recommendation = "Urgent referral - aggressive disease progression"
        elif annual_rate >= MODERATE_THRESHOLD:
            category = "MODERATE"
            recommendation = "Standard 6-month monitoring"
        elif annual_rate >= SLOW_THRESHOLD:
            category = "SLOW"
            recommendation = "Annual monitoring - typical RP progression rate"
        else:
            category = "STABLE"
            recommendation = "Possible treatment success or atypical variant"

        return {
            "progression_category":  category,
            "annual_progression_rate": round(annual_rate, 4),
            "vessel_density_change": round(vessel_change, 4),
            "pigment_change":        round(pigment_change, 4),
            "spatial_change":        round(spatial_change, 4),
            "registration_success":  success,
            "alignment_confidence":  round(confidence, 3),
            "clinical_recommendation": recommendation,
        }

    def _register_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        try:
            orb  = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
            kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
            if des1 is None or des2 is None or len(kp1) < 10:
                return img2, False, 0.0
            bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
            if len(matches) < 10:
                return img2, False, 0.0
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            if H is None:
                return img2, False, 0.0
            registered  = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
            confidence  = float(mask.sum()) / len(matches)
            return registered, True, confidence
        except Exception:
            return img2, False, 0.0

    def _compare_vessel_density(self, base: Optional[Dict], curr: Optional[Dict]) -> float:
        if not base or not curr:
            return 0.0
        return curr.get("vessel_density", 0.0) - base.get("vessel_density", 0.0)

    def _compare_pigment_clusters(self, base: Optional[Dict], curr: Optional[Dict]) -> float:
        if not base or not curr:
            return 0.0
        b_clusters = base.get("pigment_clusters", 0)
        c_clusters = curr.get("pigment_clusters", 0)
        return (c_clusters - b_clusters) / max(b_clusters, 1)

    def _compare_spatial_degradation(self, base: Optional[Dict], curr: Optional[Dict]) -> float:
        if not base or not curr:
            return 0.0
        return curr.get("spatial_degradation", 0.0) - base.get("spatial_degradation", 0.0)

def track_progression(baseline, current, months_between=12,
                      baseline_features=None, current_features=None) -> Dict:
    return ProgressionTracker().compare(baseline, current, months_between,
                                        baseline_features, current_features)
