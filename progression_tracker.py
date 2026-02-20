"""
================================================================================
PROGRESSION TRACKER - RETINAGUARD V500
================================================================================
Tracks disease progression over time by comparing serial fundus images.

FEATURES:
  1. Image Registration (align baseline vs current scan)
  2. Vessel Density Change Detection
  3. Pigmentation Progression Analysis
  4. Spatial Pattern Degradation Tracking
  5. Progression Rate Calculation (% change per year)
  6. Rapid Progression Alerts

CLINICAL SIGNIFICANCE:
  - RP is PROGRESSIVE: Diagnosis requires demonstrating change over time
  - Typical progression: 5-10% vessel loss per year
  - Rapid progression (>15% per year) = urgent referral

Version: 1.0.0
Author: RetinaGuard Development Team
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import sys

class ProgressionTracker:
    """Track retinal disease progression over serial scans"""
    
    # Progression severity thresholds (% change per year)
    RAPID_PROGRESSION_THRESHOLD = 0.15      # 15% change/year = urgent
    MODERATE_PROGRESSION_THRESHOLD = 0.08   # 8% change/year = typical RP
    SLOW_PROGRESSION_THRESHOLD = 0.03       # 3% change/year = atypical
    
    # Minimum time between scans (days) for reliable progression estimate
    MIN_SCAN_INTERVAL_DAYS = 180  # 6 months minimum
    
    def __init__(self):
        """Initialize progression tracker"""
        pass
    
    def compare_scans(self, baseline: Dict, current: Dict, 
                     baseline_date: str, current_date: str) -> Dict:
        """
        Compare baseline and current scans to detect progression
        
        Args:
            baseline: Baseline scan results (full expert panel output)
            current: Current scan results (full expert panel output)
            baseline_date: ISO date string (YYYY-MM-DD)
            current_date: ISO date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with:
                - 'time_interval_years': float (years between scans)
                - 'vessel_change': dict (density change, % per year)
                - 'pigment_change': dict (cluster count change)
                - 'spatial_change': dict (peripheral degradation change)
                - 'progression_rate': str ('RAPID', 'MODERATE', 'SLOW', 'STABLE')
                - 'progression_score': float (0-100, severity of progression)
                - 'clinical_significance': str (interpretation)
                - 'urgent': bool (requires immediate referral)
        """
        print(f"\n   [P] PROGRESSION ANALYSIS")
        print(f"      {'='*60}")
        
        # Calculate time interval
        baseline_dt = datetime.fromisoformat(baseline_date)
        current_dt = datetime.fromisoformat(current_date)
        time_delta = current_dt - baseline_dt
        years = time_delta.days / 365.25
        
        print(f"      Baseline: {baseline_date}")
        print(f"      Current:  {current_date}")
        print(f"      Interval: {time_delta.days} days ({years:.2f} years)")
        
        # Check minimum interval
        if time_delta.days < self.MIN_SCAN_INTERVAL_DAYS:
            print(f"\n      [!] WARNING: Interval too short (<{self.MIN_SCAN_INTERVAL_DAYS} days)")
            print(f"      Progression estimates unreliable - need ≥6 months between scans")
        
        # VESSEL DENSITY PROGRESSION
        vessel_change = self._analyze_vessel_progression(
            baseline.get('vessel_result', {}),
            current.get('vessel_result', {}),
            years
        )
        
        # PIGMENTATION PROGRESSION
        pigment_change = self._analyze_pigment_progression(
            baseline.get('pigment_result', {}),
            current.get('pigment_result', {}),
            years
        )
        
        # SPATIAL PATTERN PROGRESSION
        spatial_change = self._analyze_spatial_progression(
            baseline.get('spatial_result', {}),
            current.get('spatial_result', {}),
            years
        )
        
        # CALCULATE OVERALL PROGRESSION RATE
        progression_rate, progression_score = self._calculate_progression_rate(
            vessel_change, pigment_change, spatial_change
        )
        
        # CLINICAL INTERPRETATION
        clinical_sig, urgent = self._interpret_progression(
            progression_rate, progression_score, years
        )
        
        print(f"\n      [P] OVERALL PROGRESSION:")
        print(f"         Rate: {progression_rate}")
        print(f"         Score: {progression_score:.1f}/100")
        print(f"         Interpretation: {clinical_sig}")
        
        if urgent:
            print(f"         [!!] URGENT REFERRAL REQUIRED - Rapid progression detected")
        
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return {
            'time_interval_years': round(years, 2),
            'time_interval_days': time_delta.days,
            'vessel_change': vessel_change,
            'pigment_change': pigment_change,
            'spatial_change': spatial_change,
            'progression_rate': progression_rate,
            'progression_score': progression_score,
            'clinical_significance': clinical_sig,
            'urgent': urgent,
            'baseline_date': baseline_date,
            'current_date': current_date
        }
    
    def _analyze_vessel_progression(self, baseline: Dict, current: Dict, years: float) -> Dict:
        """
        Analyze vessel density change over time
        
        RP hallmark: Progressive vessel attenuation (narrowing)
        Typical: 5-10% density loss per year
        """
        baseline_density = baseline.get('density', 0.30)  # Default normal
        current_density = current.get('density', 0.30)
        
        # Absolute change
        absolute_change = current_density - baseline_density
        
        # Percent change from baseline
        if baseline_density > 0:
            percent_change = (absolute_change / baseline_density) * 100
        else:
            percent_change = 0.0
        
        # Annualized change (per year)
        if years > 0:
            change_per_year = absolute_change / years
            percent_per_year = percent_change / years
        else:
            change_per_year = 0.0
            percent_per_year = 0.0
        
        # Determine severity
        if change_per_year < -0.05:  # Loss >5% per year
            severity = 'SIGNIFICANT_LOSS'
        elif change_per_year < -0.02:  # Loss 2-5% per year
            severity = 'MODERATE_LOSS'
        elif change_per_year < 0:  # Any loss
            severity = 'MILD_LOSS'
        elif change_per_year < 0.02:  # Stable
            severity = 'STABLE'
        else:  # Increase (unusual, possible measurement error)
            severity = 'INCREASE'
        
        print(f"\n      [C] VESSEL DENSITY CHANGE:")
        print(f"         Baseline: {baseline_density:.3f} -> Current: {current_density:.3f}")
        print(f"         Change: {absolute_change:+.3f} ({percent_change:+.1f}%)")
        print(f"         Rate: {change_per_year:+.3f}/year ({percent_per_year:+.1f}%/year)")
        print(f"         Severity: {severity}")
        
        return {
            'baseline_density': baseline_density,
            'current_density': current_density,
            'absolute_change': round(absolute_change, 4),
            'percent_change': round(percent_change, 2),
            'change_per_year': round(change_per_year, 4),
            'percent_per_year': round(percent_per_year, 2),
            'severity': severity
        }
    
    def _analyze_pigment_progression(self, baseline: Dict, current: Dict, years: float) -> Dict:
        """
        Analyze bone spicule pigmentation progression
        
        RP hallmark: Progressive pigment accumulation
        """
        baseline_clusters = baseline.get('cluster_count', 0)
        current_clusters = current.get('cluster_count', 0)
        
        # Absolute change
        absolute_change = current_clusters - baseline_clusters
        
        # Percent change
        if baseline_clusters > 0:
            percent_change = (absolute_change / baseline_clusters) * 100
        else:
            if current_clusters > 0:
                percent_change = 100.0  # New pigment appearance
            else:
                percent_change = 0.0
        
        # Annualized change
        if years > 0:
            change_per_year = absolute_change / years
        else:
            change_per_year = 0.0
        
        # Determine severity
        if change_per_year > 10:  # Rapid accumulation
            severity = 'RAPID_ACCUMULATION'
        elif change_per_year > 5:  # Moderate accumulation
            severity = 'MODERATE_ACCUMULATION'
        elif change_per_year > 2:  # Slow accumulation
            severity = 'SLOW_ACCUMULATION'
        elif change_per_year > -2:  # Stable
            severity = 'STABLE'
        else:  # Reduction (atypical, measurement error likely)
            severity = 'REDUCTION'
        
        print(f"\n      [C] PIGMENTATION CHANGE:")
        print(f"         Baseline: {baseline_clusters} clusters -> Current: {current_clusters} clusters")
        print(f"         Change: {absolute_change:+.0f} ({percent_change:+.1f}%)")
        print(f"         Rate: {change_per_year:+.1f} clusters/year")
        print(f"         Severity: {severity}")
        
        return {
            'baseline_clusters': baseline_clusters,
            'current_clusters': current_clusters,
            'absolute_change': int(absolute_change),
            'percent_change': round(percent_change, 2),
            'change_per_year': round(change_per_year, 2),
            'severity': severity
        }
    
    def _analyze_spatial_progression(self, baseline: Dict, current: Dict, years: float) -> Dict:
        """
        Analyze peripheral spatial degradation progression
        
        RP hallmark: Progressive peripheral vision loss
        """
        baseline_score = baseline.get('degradation_score', 0.0)
        current_score = current.get('degradation_score', 0.0)
        
        # Absolute change (higher = worse degeneration)
        absolute_change = current_score - baseline_score
        
        # Annualized change
        if years > 0:
            change_per_year = absolute_change / years
        else:
            change_per_year = 0.0
        
        # Determine severity
        if change_per_year > 0.15:  # Rapid peripheral loss
            severity = 'RAPID_DEGENERATION'
        elif change_per_year > 0.08:  # Moderate progression
            severity = 'MODERATE_DEGENERATION'
        elif change_per_year > 0.03:  # Slow progression
            severity = 'SLOW_DEGENERATION'
        elif change_per_year > -0.03:  # Stable
            severity = 'STABLE'
        else:  # Improvement (unlikely, measurement variability)
            severity = 'IMPROVEMENT'
        
        print(f"\n      [C] SPATIAL PATTERN CHANGE:")
        print(f"         Baseline: {baseline_score:.3f} -> Current: {current_score:.3f}")
        print(f"         Change: {absolute_change:+.3f}")
        print(f"         Rate: {change_per_year:+.3f}/year")
        print(f"         Severity: {severity}")
        
        return {
            'baseline_score': baseline_score,
            'current_score': current_score,
            'absolute_change': round(absolute_change, 4),
            'change_per_year': round(change_per_year, 4),
            'severity': severity
        }
    
    def _calculate_progression_rate(self, vessel_change: Dict, 
                                   pigment_change: Dict, 
                                   spatial_change: Dict) -> Tuple[str, float]:
        """
        Calculate overall progression rate from all parameters
        
        Returns:
            Tuple of (rate_category, progression_score)
        """
        # Score each component (0-100, higher = faster progression)
        scores = []
        
        # Vessel score (weight: 40%)
        vessel_rate = abs(vessel_change['change_per_year'])
        vessel_score = min(vessel_rate * 200, 100) * 0.40
        scores.append(vessel_score)
        
        # Pigment score (weight: 30%)
        pigment_rate = pigment_change['change_per_year']
        if pigment_rate > 0:  # Only accumulation counts
            pigment_score = min(pigment_rate * 5, 100) * 0.30
        else:
            pigment_score = 0
        scores.append(pigment_score)
        
        # Spatial score (weight: 30%)
        spatial_rate = spatial_change['change_per_year']
        if spatial_rate > 0:  # Only degradation counts
            spatial_score = min(spatial_rate * 100, 100) * 0.30
        else:
            spatial_score = 0
        scores.append(spatial_score)
        
        # Combined progression score
        progression_score = sum(scores)
        
        # Categorize
        if progression_score >= 60:
            rate_category = 'RAPID'
        elif progression_score >= 30:
            rate_category = 'MODERATE'
        elif progression_score >= 10:
            rate_category = 'SLOW'
        else:
            rate_category = 'STABLE'
        
        return rate_category, progression_score
    
    def _interpret_progression(self, rate: str, score: float, years: float) -> Tuple[str, bool]:
        """
        Generate clinical interpretation
        
        Returns:
            Tuple of (interpretation_string, urgent_flag)
        """
        interpretations = {
            'RAPID': (
                f"Rapid disease progression detected ({score:.0f}/100 severity). "
                f"Significant worsening over {years:.1f} years. URGENT referral to retina specialist required.",
                True
            ),
            'MODERATE': (
                f"Moderate progression ({score:.0f}/100 severity). "
                f"Typical RP progression pattern over {years:.1f} years. Continue monitoring every 6-12 months.",
                False
            ),
            'SLOW': (
                f"Slow progression ({score:.0f}/100 severity). "
                f"Mild changes over {years:.1f} years. Monitor annually.",
                False
            ),
            'STABLE': (
                f"Disease appears stable ({score:.0f}/100 severity). "
                f"No significant progression over {years:.1f} years. Continue routine monitoring.",
                False
            )
        }
        
        return interpretations.get(rate, ("Unknown progression rate", False))
    
    def register_images(self, baseline_image: np.ndarray, 
                       current_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register (align) baseline and current images for accurate comparison
        
        Uses feature-based registration (ORB + RANSAC)
        
        Args:
            baseline_image: Reference image
            current_image: Image to align
            
        Returns:
            Tuple of (aligned_baseline, aligned_current)
        """
        print(f"\n      🔄 IMAGE REGISTRATION:")
        
        # Convert to grayscale
        if len(baseline_image.shape) == 3:
            baseline_gray = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)
        else:
            baseline_gray = baseline_image
            
        if len(current_image.shape) == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_image
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=5000)
        
        kp1, des1 = orb.detectAndCompute(baseline_gray, None)
        kp2, des2 = orb.detectAndCompute(current_gray, None)
        
        print(f"         Baseline features: {len(kp1)}")
        print(f"         Current features: {len(kp2)}")
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        print(f"         Good matches: {len(good_matches)}")
        
        if len(good_matches) < 10:
            print(f"         [!] WARNING: Insufficient matches for reliable registration")
            return baseline_image, current_image
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography (perspective transform)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Warp current image to align with baseline
            h, w = baseline_gray.shape
            aligned_current = cv2.warpPerspective(current_image, H, (w, h))
            
            inliers = np.sum(mask)
            print(f"         Registration successful: {inliers} inliers")
            print(f"         Alignment quality: {'GOOD' if inliers > 50 else 'FAIR' if inliers > 20 else 'POOR'}")
            
            return baseline_image, aligned_current
        else:
            print(f"         [X] Registration failed - using original images")
            return baseline_image, current_image


# Convenience function for external use
def track_progression(baseline_data: Dict, current_data: Dict, 
                     baseline_date: str, current_date: str) -> Dict:
    """
    Compare two scans to detect disease progression
    
    Args:
        baseline_data: Baseline expert panel results
        current_data: Current expert panel results
        baseline_date: ISO date (YYYY-MM-DD)
        current_date: ISO date (YYYY-MM-DD)
        
    Returns:
        Progression analysis results
    """
    tracker = ProgressionTracker()
    return tracker.compare_scans(baseline_data, current_data, baseline_date, current_date)


# Testing harness
if __name__ == "__main__":
    print("="*80)
    print("PROGRESSION TRACKER - TEST SUITE")
    print("="*80)
    
    # Simulate baseline scan (1 year ago)
    baseline = {
        'vessel_result': {'density': 0.28, 'severity': 'MILD'},
        'pigment_result': {'cluster_count': 12, 'severity': 'MILD'},
        'spatial_result': {'degradation_score': 0.25, 'severity': 'MILD'}
    }
    
    # Test Case 1: Rapid progression
    current_rapid = {
        'vessel_result': {'density': 0.18, 'severity': 'MODERATE'},
        'pigment_result': {'cluster_count': 28, 'severity': 'MODERATE'},
        'spatial_result': {'degradation_score': 0.45, 'severity': 'MODERATE'}
    }
    
    print("\n[TEST 1] Rapid Progression (1 year):")
    result1 = track_progression(baseline, current_rapid, '2025-02-19', '2026-02-19')
    print(f"Rate: {result1['progression_rate']} | Score: {result1['progression_score']:.1f}")
    print(f"Urgent: {result1['urgent']}")
    
    # Test Case 2: Stable disease
    current_stable = {
        'vessel_result': {'density': 0.27, 'severity': 'MILD'},
        'pigment_result': {'cluster_count': 13, 'severity': 'MILD'},
        'spatial_result': {'degradation_score': 0.26, 'severity': 'MILD'}
    }
    
    print("\n[TEST 2] Stable Disease (1 year):")
    result2 = track_progression(baseline, current_stable, '2025-02-19', '2026-02-19')
    print(f"Rate: {result2['progression_rate']} | Score: {result2['progression_score']:.1f}")
    print(f"Urgent: {result2['urgent']}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
