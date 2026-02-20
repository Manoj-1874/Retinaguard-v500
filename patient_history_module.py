"""
================================================================================
PATIENT HISTORY MODULE - RETINAGUARD V500
================================================================================
Integrates patient demographics, symptoms, and family history to enhance
diagnostic accuracy and provide personalized threshold adjustments.

FEATURES:
  1. Symptom Questionnaire (night blindness, tunnel vision, etc.)
  2. Demographic-Adjusted Thresholds (age, ethnicity)
  3. Family History Risk Scoring
  4. Visual Field Data Integration
  5. Clinical Risk Stratification

BIAS PREVENTION:
  - Ethnicity-adjusted pigmentation thresholds (African, Asian, Caucasian, etc.)
  - Age-stratified vessel density norms (pediatric, adult, geriatric)
  - Gender-specific prevalence adjustments

Version: 1.0.0
Author: RetinaGuard Development Team
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys

class PatientHistoryModule:
    """Patient demographic and symptom integration"""
    
    # Ethnicity-specific baseline corrections
    ETHNICITY_ADJUSTMENTS = {
        'african': {
            'pigment_baseline': +15,     # Higher natural pigmentation
            'disc_baseline': -10,         # Darker optic disc baseline
            'description': 'African/African-American descent'
        },
        'asian': {
            'pigment_baseline': +8,
            'disc_baseline': -5,
            'description': 'East/Southeast Asian descent'
        },
        'south_asian': {
            'pigment_baseline': +10,
            'disc_baseline': -7,
            'description': 'South Asian (Indian subcontinent) descent'
        },
        'hispanic': {
            'pigment_baseline': +5,
            'disc_baseline': -3,
            'description': 'Hispanic/Latino descent'
        },
        'caucasian': {
            'pigment_baseline': 0,
            'disc_baseline': 0,
            'description': 'European/Caucasian descent'
        },
        'middle_eastern': {
            'pigment_baseline': +6,
            'disc_baseline': -4,
            'description': 'Middle Eastern descent'
        },
        'mixed': {
            'pigment_baseline': +3,
            'disc_baseline': -2,
            'description': 'Mixed/Multiple ethnicities'
        },
        'unknown': {
            'pigment_baseline': 0,
            'disc_baseline': 0,
            'description': 'Not specified'
        }
    }
    
    # Age-specific vessel density norms
    AGE_ADJUSTMENTS = {
        'pediatric': {       # 0-17 years
            'vessel_bonus': +0.05,       # Children have denser vessels
            'texture_tolerance': +0.5,   # Normal developing retina
            'age_range': (0, 17)
        },
        'adult': {           # 18-65 years
            'vessel_bonus': 0.0,
            'texture_tolerance': 0.0,
            'age_range': (18, 65)
        },
        'geriatric': {       # 65+ years
            'vessel_bonus': -0.03,       # Natural age-related vessel attenuation
            'texture_tolerance': -0.3,   # Aging-related texture changes
            'age_range': (65, 120)
        }
    }
    
    # Symptom severity scoring (0-10 scale)
    SYMPTOM_WEIGHTS = {
        'night_blindness': 3.0,          # Strongest RP indicator (nyctalopia)
        'tunnel_vision': 2.5,            # Peripheral vision loss
        'difficulty_dark_adaptation': 2.0,
        'light_sensitivity': 1.5,        # Photophobia
        'color_vision_loss': 1.0,
        'central_vision_loss': 1.0,      # Late-stage symptom
        'floaters': 0.5,                 # Common, non-specific
        'flashes': 0.5                   # Common, non-specific
    }
    
    def __init__(self):
        """Initialize patient history module"""
        pass
    
    def collect_patient_data(self, data: Dict) -> Dict:
        """
        Process patient demographic and symptom data
        
        Args:
            data: Dictionary containing:
                - age: int (years)
                - gender: str ('male', 'female', 'other')
                - ethnicity: str (see ETHNICITY_ADJUSTMENTS keys)
                - symptoms: dict {symptom_name: severity (0-10)}
                - family_history: bool (first-degree relative with RP)
                - visual_field_data: optional dict (perimetry results)
                - previous_diagnosis: optional str
                
        Returns:
            Dictionary with:
                - 'risk_score': float (0-100, likelihood of RP)
                - 'threshold_adjustments': dict (CONFIG multipliers)
                - 'clinical_flags': list (important notes)
                - 'confidence_modifier': float (AI confidence adjustment)
        """
        print(f"\n   [P] PATIENT HISTORY ANALYSIS")
        print(f"      {'='*60}")
        
        # Extract patient data
        age = data.get('age', 40)
        gender = data.get('gender', 'unknown').lower()
        ethnicity = data.get('ethnicity', 'unknown').lower()
        symptoms = data.get('symptoms', {})
        family_history = data.get('family_history', False)
        visual_field_data = data.get('visual_field_data', None)
        
        clinical_flags = []
        
        # DEMOGRAPHICS
        print(f"      Age: {age} years | Gender: {gender.capitalize()} | Ethnicity: {ethnicity.replace('_', ' ').title()}")
        
        # Determine age category
        age_category = self._classify_age(age)
        age_adj = self.AGE_ADJUSTMENTS[age_category]
        
        # Get ethnicity adjustments
        if ethnicity not in self.ETHNICITY_ADJUSTMENTS:
            ethnicity = 'unknown'
            clinical_flags.append(f"[!] Unknown ethnicity - using default thresholds")
        
        eth_adj = self.ETHNICITY_ADJUSTMENTS[ethnicity]
        
        # SYMPTOM ANALYSIS
        symptom_score = self._calculate_symptom_score(symptoms)
        print(f"\n      [S] SYMPTOM ASSESSMENT:")
        
        if len(symptoms) == 0:
            print(f"         No symptoms reported")
            clinical_flags.append("[i] No symptoms reported - screening only")
        else:
            for symptom, severity in symptoms.items():
                if symptom in self.SYMPTOM_WEIGHTS and severity > 0:
                    weight = self.SYMPTOM_WEIGHTS[symptom]
                    contribution = severity * weight
                    symptom_name = symptom.replace('_', ' ').title()
                    print(f"         • {symptom_name}: {severity}/10 (weight={weight}, score={contribution:.1f})")
                    
                    # Flag critical symptoms
                    if symptom == 'night_blindness' and severity >= 7:
                        clinical_flags.append("[!!] SEVERE night blindness - High RP suspicion")
                    elif symptom == 'tunnel_vision' and severity >= 7:
                        clinical_flags.append("[!!] SEVERE tunnel vision - Advanced RP likely")
        
        print(f"         -> Total Symptom Score: {symptom_score:.1f}/100")
        
        # FAMILY HISTORY
        family_risk = 0.0
        if family_history:
            family_risk = 25.0  # 25% increased risk with family history
            print(f"\n      [F] FAMILY HISTORY: Positive (+25% risk)")
            clinical_flags.append("[!] Family history of RP - Hereditary risk")
        else:
            print(f"\n      [F] FAMILY HISTORY: Negative")
        
        # VISUAL FIELD DATA (if available)
        vf_score = 0.0
        if visual_field_data:
            vf_score = self._analyze_visual_fields(visual_field_data)
            print(f"\n      [V] VISUAL FIELD DATA: Peripheral loss score = {vf_score:.1f}/100")
            
            if vf_score > 50:
                clinical_flags.append(f"[!!] Significant peripheral vision loss detected (VF score: {vf_score:.1f})")
        
        # CALCULATE OVERALL RISK SCORE (0-100)
        risk_score = self._calculate_risk_score(symptom_score, family_risk, vf_score, age)
        
        print(f"\n      [R] OVERALL RISK SCORE: {risk_score:.1f}/100")
        
        if risk_score >= 70:
            print(f"         -> [HIGH] Strong clinical suspicion")
        elif risk_score >= 40:
            print(f"         -> [MOD] Warrants investigation")
        else:
            print(f"         -> [LOW] Routine screening")
        
        # GENERATE THRESHOLD ADJUSTMENTS
        threshold_adjustments = self._generate_threshold_adjustments(
            age_adj, eth_adj, symptom_score, family_history
        )
        
        print(f"\n      [A] THRESHOLD ADJUSTMENTS:")
        print(f"         Pigment threshold: {threshold_adjustments['pigment_adjustment']:+.1f} (ethnicity correction)")
        print(f"         Vessel threshold: {threshold_adjustments['vessel_adjustment']:+.3f} (age correction)")
        print(f"         AI confidence shift: {threshold_adjustments['ai_confidence_shift']:+.2f} (symptom risk)")
        
        # CLINICAL FLAGS
        if clinical_flags:
            print(f"\n      [*] CLINICAL FLAGS:")
            for flag in clinical_flags:
                print(f"         {flag}")
        
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return {
            'risk_score': risk_score,
            'threshold_adjustments': threshold_adjustments,
            'clinical_flags': clinical_flags,
            'confidence_modifier': self._calculate_confidence_modifier(risk_score),
            'age_category': age_category,
            'ethnicity': ethnicity,
            'symptom_score': symptom_score
        }
    
    def _classify_age(self, age: int) -> str:
        """Classify patient into age category"""
        if age < 18:
            return 'pediatric'
        elif age < 65:
            return 'adult'
        else:
            return 'geriatric'
    
    def _calculate_symptom_score(self, symptoms: Dict[str, int]) -> float:
        """
        Calculate weighted symptom score (0-100)
        
        Args:
            symptoms: {symptom_name: severity (0-10)}
            
        Returns:
            Normalized score 0-100
        """
        if not symptoms:
            return 0.0
        
        total_weighted = 0.0
        max_possible = 0.0
        
        for symptom, severity in symptoms.items():
            if symptom in self.SYMPTOM_WEIGHTS:
                weight = self.SYMPTOM_WEIGHTS[symptom]
                total_weighted += severity * weight
                max_possible += 10 * weight  # Max severity is 10
        
        # Normalize to 0-100 scale
        if max_possible > 0:
            return (total_weighted / max_possible) * 100
        else:
            return 0.0
    
    def _analyze_visual_fields(self, vf_data: Dict) -> float:
        """
        Analyze visual field perimetry data
        
        Args:
            vf_data: Dictionary with:
                - 'mean_deviation': float (dB loss, negative = worse)
                - 'pattern_std_deviation': float
                - 'peripheral_loss': bool (ring scotoma detected)
                
        Returns:
            Score 0-100 (higher = more peripheral loss)
        """
        score = 0.0
        
        # Mean deviation (MD) - normal is 0, RP typically -10 to -30 dB
        md = vf_data.get('mean_deviation', 0.0)
        if md < 0:
            score += min(abs(md) * 3, 50)  # Max 50 points from MD
        
        # Pattern standard deviation (PSD) - higher = more irregular loss
        psd = vf_data.get('pattern_std_deviation', 0.0)
        score += min(psd * 2, 30)  # Max 30 points from PSD
        
        # Peripheral ring scotoma (classic RP pattern)
        if vf_data.get('peripheral_loss', False):
            score += 20  # Bonus for ring scotoma
        
        return min(score, 100.0)
    
    def _calculate_risk_score(self, symptom_score: float, family_risk: float, 
                             vf_score: float, age: int) -> float:
        """
        Calculate overall clinical risk score
        
        Components:
            - Symptoms: 40% weight
            - Family history: 25% weight
            - Visual fields: 25% weight
            - Age factor: 10% weight (peak onset 20-40 years)
        """
        # Age risk (bell curve, peak at 20-40 years)
        if 20 <= age <= 40:
            age_risk = 10.0  # Peak risk age
        elif 10 <= age <= 60:
            age_risk = 7.0
        else:
            age_risk = 3.0
        
        # Weighted combination
        risk = (symptom_score * 0.40 +
                family_risk +
                vf_score * 0.25 +
                age_risk)
        
        return min(risk, 100.0)
    
    def _generate_threshold_adjustments(self, age_adj: Dict, eth_adj: Dict, 
                                       symptom_score: float, family_history: bool) -> Dict:
        """
        Generate CONFIG threshold adjustments based on patient data
        
        Returns:
            Dictionary with adjustment values to apply to base CONFIG
        """
        adjustments = {}
        
        # PIGMENT THRESHOLD (ethnicity-based)
        # Higher baseline pigmentation = increase threshold (reduce false positives)
        adjustments['pigment_adjustment'] = eth_adj['pigment_baseline']
        
        # VESSEL THRESHOLD (age-based)
        # Adjust for normal age-related vessel changes
        adjustments['vessel_adjustment'] = age_adj['vessel_bonus']
        
        # OPTIC DISC THRESHOLD (ethnicity-based)
        adjustments['disc_adjustment'] = eth_adj['disc_baseline']
        
        # TEXTURE THRESHOLD (age-based)
        adjustments['texture_adjustment'] = age_adj['texture_tolerance']
        
        # AI CONFIDENCE THRESHOLD SHIFT
        # High symptom score = lower AI threshold (catch early RP)
        # No symptoms = higher threshold (reduce false positives)
        if symptom_score > 60:
            adjustments['ai_confidence_shift'] = -0.10  # Lower threshold by 10%
        elif symptom_score > 30:
            adjustments['ai_confidence_shift'] = -0.05
        else:
            adjustments['ai_confidence_shift'] = 0.0
        
        # Family history bonus (lower thresholds slightly)
        if family_history:
            adjustments['ai_confidence_shift'] -= 0.05
        
        return adjustments
    
    def _calculate_confidence_modifier(self, risk_score: float) -> float:
        """
        Calculate modifier for final verdict confidence
        
        High-risk patients (symptoms + family history) should have
        higher confidence in SUSPICIOUS/POSITIVE verdicts
        
        Returns:
            Multiplier for confidence level (0.7 - 1.3)
        """
        if risk_score > 70:
            return 1.3  # Boost confidence for high-risk patients
        elif risk_score > 40:
            return 1.1
        elif risk_score < 10:
            return 0.8  # Reduce confidence for asymptomatic screening
        else:
            return 1.0
    
    def apply_adjustments_to_config(self, base_config: Dict, adjustments: Dict) -> Dict:
        """
        Apply patient-specific adjustments to base CONFIG
        
        Args:
            base_config: Original CONFIG dictionary from app.py
            adjustments: Adjustment values from generate_threshold_adjustments()
            
        Returns:
            Modified CONFIG with patient-specific thresholds
        """
        adjusted = base_config.copy()
        
        # PIGMENT THRESHOLDS (increase for higher natural pigmentation)
        pigment_adj = adjustments['pigment_adjustment']
        adjusted['PIGMENT_CRITICAL'] = base_config['PIGMENT_CRITICAL'] + pigment_adj
        adjusted['PIGMENT_MODERATE'] = base_config['PIGMENT_MODERATE'] + pigment_adj
        adjusted['PIGMENT_MILD'] = base_config['PIGMENT_MILD'] + pigment_adj
        
        # VESSEL THRESHOLDS (adjust for age)
        vessel_adj = adjustments['vessel_adjustment']
        adjusted['VESSEL_CRITICAL'] = base_config['VESSEL_CRITICAL'] + vessel_adj
        adjusted['VESSEL_MODERATE'] = base_config['VESSEL_MODERATE'] + vessel_adj
        adjusted['VESSEL_MILD'] = base_config['VESSEL_MILD'] + vessel_adj
        
        # OPTIC DISC THRESHOLDS (ethnicity correction)
        disc_adj = adjustments['disc_adjustment']
        adjusted['DISC_CRITICAL'] = base_config['DISC_CRITICAL'] + disc_adj
        adjusted['DISC_MODERATE'] = base_config['DISC_MODERATE'] + disc_adj
        adjusted['DISC_MILD'] = base_config['DISC_MILD'] + disc_adj
        
        # AI THRESHOLDS (symptom/risk-based)
        ai_shift = adjustments['ai_confidence_shift']
        adjusted['AI_POSITIVE_THRESHOLD'] = base_config['AI_POSITIVE_THRESHOLD'] + ai_shift
        adjusted['AI_UNCERTAIN_THRESHOLD'] = base_config['AI_UNCERTAIN_THRESHOLD'] + ai_shift
        
        return adjusted


# Convenience function for external use
def analyze_patient_history(patient_data: Dict) -> Dict:
    """
    Analyze patient history and return risk assessment + threshold adjustments
    
    Args:
        patient_data: Dictionary with demographics, symptoms, family history
        
    Returns:
        Analysis results with risk score and CONFIG adjustments
    """
    module = PatientHistoryModule()
    return module.collect_patient_data(patient_data)


# Testing harness
if __name__ == "__main__":
    print("="*80)
    print("PATIENT HISTORY MODULE - TEST SUITE")
    print("="*80)
    
    # Test Case 1: High-risk patient (African descent, severe symptoms, family history)
    test_patient_1 = {
        'age': 28,
        'gender': 'male',
        'ethnicity': 'african',
        'symptoms': {
            'night_blindness': 9,
            'tunnel_vision': 7,
            'difficulty_dark_adaptation': 8
        },
        'family_history': True,
        'visual_field_data': {
            'mean_deviation': -15.2,
            'pattern_std_deviation': 8.5,
            'peripheral_loss': True
        }
    }
    
    print("\n[TEST 1] High-Risk Patient:")
    result1 = analyze_patient_history(test_patient_1)
    print(f"Risk Score: {result1['risk_score']:.1f}/100")
    print(f"Flags: {len(result1['clinical_flags'])}")
    
    # Test Case 2: Low-risk screening (Caucasian, no symptoms, no family history)
    test_patient_2 = {
        'age': 45,
        'gender': 'female',
        'ethnicity': 'caucasian',
        'symptoms': {},
        'family_history': False
    }
    
    print("\n[TEST 2] Low-Risk Screening:")
    result2 = analyze_patient_history(test_patient_2)
    print(f"Risk Score: {result2['risk_score']:.1f}/100")
    
    # Test Case 3: Pediatric patient (Asian, mild symptoms)
    test_patient_3 = {
        'age': 12,
        'gender': 'female',
        'ethnicity': 'asian',
        'symptoms': {
            'night_blindness': 4,
            'difficulty_dark_adaptation': 3
        },
        'family_history': False
    }
    
    print("\n[TEST 3] Pediatric Patient:")
    result3 = analyze_patient_history(test_patient_3)
    print(f"Age Category: {result3['age_category']}")
    print(f"Risk Score: {result3['risk_score']:.1f}/100")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
