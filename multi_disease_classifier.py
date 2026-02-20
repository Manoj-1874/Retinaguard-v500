"""
================================================================================
MULTI-DISEASE CLASSIFIER - RETINAGUARD V500
================================================================================
Differential diagnosis module for detecting multiple retinal diseases,
not just Retinitis Pigmentosa.

SUPPORTED DISEASES (7 CONDITIONS):
  1. Retinitis Pigmentosa (RP) - Primary target
  2. Diabetic Retinopathy (DR) - Microaneurysms, hemorrhages
  3. Age-Related Macular Degeneration (AMD) - Drusen, geographic atrophy
  4. Glaucoma - Optic disc cupping, RNFL loss
  5. Hypertensive Retinopathy - AV nicking, cotton-wool spots
  6. Choroideremia - Progressive chorioretinal atrophy
  7. Usher Syndrome - RP + hearing loss (systemic)

FEATURE-BASED CLASSIFICATION:
  - Uses same expert scanner outputs as RP detection
  - Disease-specific pattern recognition
  - Confidence scoring for each disease
  - Top differential diagnoses ranked

Version: 1.0.0
Author: RetinaGuard Development Team
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple
import sys

class MultiDiseaseClassifier:
    """Differential diagnosis for retinal diseases"""
    
    # Disease-specific feature patterns
    DISEASE_PATTERNS = {
        'retinitis_pigmentosa': {
            'name': 'Retinitis Pigmentosa',
            'key_features': {
                'bone_spicules': 0.35,      # Strongest indicator
                'vessel_attenuation': 0.30,
                'optic_disc_pallor': 0.25,
                'peripheral_loss': 0.10
            },
            'exclusions': ['microaneurysms', 'drusen', 'disc_cupping']
        },
        'diabetic_retinopathy': {
            'name': 'Diabetic Retinopathy',
            'key_features': {
                'microaneurysms': 0.35,
                'hemorrhages': 0.30,
                'exudates': 0.20,
                'cotton_wool_spots': 0.10,
                'neovascularization': 0.05
            },
            'exclusions': ['bone_spicules', 'disc_pallor']
        },
        'amd': {
            'name': 'Age-Related Macular Degeneration',
            'key_features': {
                'drusen': 0.40,
                'geographic_atrophy': 0.30,
                'macular_edema': 0.20,
                'abnormal_texture': 0.10
            },
            'exclusions': ['bone_spicules', 'vessel_attenuation']
        },
        'glaucoma': {
            'name': 'Glaucoma',
            'key_features': {
                'disc_cupping': 0.45,
                'rnfl_thinning': 0.35,
                'disc_pallor': 0.15,
                'peripapillary_atrophy': 0.05
            },
            'exclusions': ['bone_spicules', 'microaneurysms']
        },
        'hypertensive_retinopathy': {
            'name': 'Hypertensive Retinopathy',
            'key_features': {
                'av_nicking': 0.30,
                'vessel_tortuosity': 0.25,
                'cotton_wool_spots': 0.20,
                'hemorrhages': 0.15,
                'optic_disc_edema': 0.10
            },
            'exclusions': ['bone_spicules', 'drusen']
        },
        'choroideremia': {
            'name': 'Choroideremia',
            'key_features': {
                'chorioretinal_atrophy': 0.40,
                'peripheral_loss': 0.30,
                'vessel_attenuation': 0.20,
                'macular_preservation': 0.10  # Central vision spared until late
            },
            'exclusions': ['bone_spicules', 'microaneurysms']
        },
        'usher_syndrome': {
            'name': 'Usher Syndrome (RP variant)',
            'key_features': {
                'bone_spicules': 0.30,
                'vessel_attenuation': 0.25,
                'disc_pallor': 0.20,
                'peripheral_loss': 0.15,
                'early_onset': 0.10  # Requires patient history
            },
            'exclusions': ['drusen', 'microaneurysms']
        }
    }
    
    def __init__(self):
        """Initialize multi-disease classifier"""
        pass
    
    def classify(self, expert_results: Dict, patient_age: int = 40, 
                patient_history: Dict = None) -> Dict:
        """
        Perform differential diagnosis based on expert scann results
        
        Args:
            expert_results: Dictionary of all 10 expert scanner outputs
            patient_age: Patient age (for age-related diseases)
            patient_history: Optional patient history data
            
        Returns:
            Dictionary with:
                - 'top_diagnosis': str (most likely disease)
                - 'differential': list (all diseases with confidence >10%)
                - 'disease_scores': dict (all disease confidence scores)
                - 'clinical_notes': list (important observations)
        """
        print(f"\n   [D] DIFFERENTIAL DIAGNOSIS")
        print(f"      {'='*60}")
        
        # Extract feature vector from expert results
        features = self._extract_features(expert_results, patient_age, patient_history)
        
        print(f"\n      [F] FEATURE EXTRACTION:")
        for feature, value in features.items():
            if value > 0:
                print(f"         • {feature.replace('_', ' ').title()}: {value:.2f}")
        
        # Calculate confidence score for each disease
        disease_scores = {}
        for disease_id, pattern in self.DISEASE_PATTERNS.items():
            score = self._calculate_disease_score(features, pattern)
            disease_scores[disease_id] = score
        
        # Sort diseases by confidence (descending)
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top diagnosis
        top_disease_id, top_score = sorted_diseases[0]
        top_diagnosis = self.DISEASE_PATTERNS[top_disease_id]['name']
        
        # Differential diagnoses (confidence >10%)
        differential = []
        for disease_id, score in sorted_diseases:
            if score >= 0.10:  # 10% threshold
                disease_name = self.DISEASE_PATTERNS[disease_id]['name']
                differential.append({
                    'disease': disease_name,
                    'disease_id': disease_id,
                    'confidence': round(score * 100, 1)
                })
        
        # Clinical notes
        clinical_notes = self._generate_clinical_notes(
            sorted_diseases, features, patient_age
        )
        
        # Display results
        print(f"\n      [D] DIFFERENTIAL DIAGNOSIS:")
        for i, diag in enumerate(differential[:5], 1):  # Top 5
            rank_symbol = "[1]" if i == 1 else "[2]" if i == 2 else "[3]" if i == 3 else f"[{i}]"
            print(f"         {rank_symbol} {diag['disease']}: {diag['confidence']}%")
        
        if clinical_notes:
            print(f"\n      [N] CLINICAL NOTES:")
            for note in clinical_notes:
                print(f"         - {note}")
        
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return {
            'top_diagnosis': top_diagnosis,
            'top_confidence': round(top_score * 100, 1),
            'differential': differential,
            'disease_scores': {k: round(v * 100, 1) for k, v in disease_scores.items()},
            'clinical_notes': clinical_notes
        }
    
    def _extract_features(self, expert_results: Dict, patient_age: int,
                         patient_history: Dict = None) -> Dict[str, float]:
        """
        Extract disease-relevant features from expert scanner results
        
        Returns:
            Dictionary of normalized feature values (0-1 scale)
        """
        features = {}
        
        # RETINITIS PIGMENTOSA features
        pigment = expert_results.get('pigment_result', {})
        features['bone_spicules'] = min(pigment.get('cluster_count', 0) / 30.0, 1.0)
        
        vessel = expert_results.get('vessel_result', {})
        vessel_density = vessel.get('density', 0.30)
        features['vessel_attenuation'] = max(0, (0.30 - vessel_density) / 0.30)  # Lower = more attenuation
        
        disc = expert_results.get('optic_disc_result', {})
        disc_brightness = disc.get('brightness', 160)
        features['optic_disc_pallor'] = max(0, (disc_brightness - 180) / 50)  # >180 = pallor
        
        spatial = expert_results.get('spatial_result', {})
        features['peripheral_loss'] = spatial.get('degradation_score', 0.0)
        
        # DIABETIC RETINOPATHY features (simplified - needs dedicated detectors)
        # For now, use texture and bright lesions as proxies
        texture = expert_results.get('texture_result', {})
        features['microaneurysms'] = 0.0  # Placeholder - needs dedicated detector
        features['hemorrhages'] = 0.0     # Placeholder
        features['exudates'] = min(texture.get('local_variation', 0) / 10.0, 1.0)
        features['cotton_wool_spots'] = 0.0  # Placeholder
        features['neovascularization'] = 0.0  # Placeholder
        
        # AMD features
        bright_lesion = expert_results.get('bright_lesion_result', {})
        features['drusen'] = min(bright_lesion.get('fleck_count', 0) / 20.0, 1.0)
        
        macula = expert_results.get('macula_result', {})
        features['macular_edema'] = macula.get('cme_score', 0.0)
        features['abnormal_texture'] = min(texture.get('entropy', 5.0) / 7.0, 1.0)
        features['geographic_atrophy'] = 0.0  # Placeholder
        
        # GLAUCOMA features
        # High disc brightness + specific shape = cupping
        features['disc_cupping'] = 0.0  # Placeholder - needs cup/disc ratio
        features['rnfl_thinning'] = features['vessel_attenuation'] * 0.5  # Proxy
        features['peripapillary_atrophy'] = 0.0  # Placeholder
        
        # HYPERTENSIVE RETINOPATHY features
        tortuosity = expert_results.get('tortuosity_result', {})
        tortuous_vessels = tortuosity.get('tortuosity', 1.0)
        features['vessel_tortuosity'] = max(0, (tortuous_vessels - 1.3) / 1.0)
        features['av_nicking'] = 0.0  # Placeholder
        features['optic_disc_edema'] = max(0, (170 - disc_brightness) / 30)  # Darker disc
        
        # CHOROIDEREMIA features
        features['chorioretinal_atrophy'] = features['peripheral_loss'] * 0.8
        features['macular_preservation'] = max(0, 1.0 - features['macular_edema'])
        
        # USHER SYNDROME (similar to RP but earlier onset)
        features['early_onset'] = 1.0 if patient_age < 20 else 0.5 if patient_age < 30 else 0.0
        
        return features
    
    def _calculate_disease_score(self, features: Dict[str, float], 
                                 pattern: Dict) -> float:
        """
        Calculate disease confidence score based on feature matching
        
        Args:
            features: Extracted feature values
            pattern: Disease-specific pattern definition
            
        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        
        # Positive evidence (weighted sum of matching features)
        for feature_name, weight in pattern['key_features'].items():
            if feature_name in features:
                score += features[feature_name] * weight
        
        # Negative evidence (presence of exclusionary features)
        exclusions = pattern.get('exclusions', [])
        for exclusion in exclusions:
            if exclusion in features and features[exclusion] > 0.3:
                score *= 0.5  # Penalize if exclusionary feature present
        
        return min(score, 1.0)
    
    def _generate_clinical_notes(self, sorted_diseases: List[Tuple], 
                                features: Dict, patient_age: int) -> List[str]:
        """
        Generate clinical interpretation notes
        
        Returns:
            List of clinical observation strings
        """
        notes = []
        
        # Age-related observations
        if patient_age >= 50:
            if features.get('drusen', 0) > 0.3 or features.get('macular_edema', 0) > 0.3:
                notes.append("Age >50: Consider AMD as differential")
        
        if patient_age < 20:
            if features.get('bone_spicules', 0) > 0.3:
                notes.append("Early onset RP: Consider Usher syndrome (genetic testing + audiology)")
        
        # Feature-specific notes
        if features.get('bone_spicules', 0) > 0.5 and features.get('vessel_attenuation', 0) < 0.3:
            notes.append("Isolated pigmentation without vessel changes: Consider benign causes (CHRPE, laser scars)")
        
        if features.get('vessel_attenuation', 0) > 0.5 and features.get('bone_spicules', 0) < 0.2:
            notes.append("Vessel attenuation without pigment: Consider RP Sine Pigmento variant")
        
        if features.get('macular_edema', 0) > 0.5:
            notes.append("Macular edema detected: OCT recommended for CME confirmation")
        
        # Differential diagnosis ambiguity
        top_score = sorted_diseases[0][1]
        second_score = sorted_diseases[1][1] if len(sorted_diseases) > 1 else 0
        
        if abs(top_score - second_score) < 0.15:
            top_name = self.DISEASE_PATTERNS[sorted_diseases[0][0]]['name']
            second_name = self.DISEASE_PATTERNS[sorted_diseases[1][0]]['name']
            notes.append(f"Ambiguous diagnosis: {top_name} vs {second_name} - clinical correlation required")
        
        return notes


# Convenience function for external use
def classify_diseases(expert_results: Dict, patient_age: int = 40, 
                     patient_history: Dict = None) -> Dict:
    """
    Perform multi-disease differential diagnosis
    
    Args:
        expert_results: Expert scanner panel results
        patient_age: Patient age
        patient_history: Optional patient history
        
    Returns:
        Differential diagnosis results
    """
    classifier = MultiDiseaseClassifier()
    return classifier.classify(expert_results, patient_age, patient_history)


# Testing harness
if __name__ == "__main__":
    print("="*80)
    print("MULTI-DISEASE CLASSIFIER - TEST SUITE")
    print("="*80)
    
    # Test Case 1: Classic RP
    rp_results = {
        'pigment_result': {'cluster_count': 35, 'severity': 'CRITICAL'},
        'vessel_result': {'density': 0.18, 'severity': 'MODERATE'},
        'optic_disc_result': {'brightness': 205, 'severity': 'MODERATE'},
        'spatial_result': {'degradation_score': 0.5, 'severity': 'MODERATE'},
        'texture_result': {'entropy': 6.0, 'local_variation': 3.5},
        'bright_lesion_result': {'fleck_count': 0},
        'macula_result': {'cme_score': 0.1},
        'tortuosity_result': {'tortuosity': 1.1}
    }
    
    print("\n[TEST 1] Classic Retinitis Pigmentosa:")
    result1 = classify_diseases(rp_results, patient_age=35)
    print(f"Top Diagnosis: {result1['top_diagnosis']} ({result1['top_confidence']}%)")
    
    # Test Case 2: AMD (older patient, drusen)
    amd_results = {
        'pigment_result': {'cluster_count': 2, 'severity': 'NORMAL'},
        'vessel_result': {'density': 0.28, 'severity': 'NORMAL'},
        'optic_disc_result': {'brightness': 165, 'severity': 'NORMAL'},
        'spatial_result': {'degradation_score': 0.2, 'severity': 'NORMAL'},
        'texture_result': {'entropy': 6.5, 'local_variation': 5.0},
        'bright_lesion_result': {'fleck_count': 25},  # Drusen
        'macula_result': {'cme_score': 0.6},
        'tortuosity_result': {'tortuosity': 1.0}
    }
    
    print("\n[TEST 2] Age-Related Macular Degeneration:")
    result2 = classify_diseases(amd_results, patient_age=72)
    print(f"Top Diagnosis: {result2['top_diagnosis']} ({result2['top_confidence']}%)")
    
    # Test Case 3: Hypertensive Retinopathy
    htn_results = {
        'pigment_result': {'cluster_count': 0, 'severity': 'NORMAL'},
        'vessel_result': {'density': 0.32, 'severity': 'NORMAL'},
        'optic_disc_result': {'brightness': 155, 'severity': 'NORMAL'},
        'spatial_result': {'degradation_score': 0.1, 'severity': 'NORMAL'},
        'texture_result': {'entropy': 5.8, 'local_variation': 3.0},
        'bright_lesion_result': {'fleck_count': 0},
        'macula_result': {'cme_score': 0.0},
        'tortuosity_result': {'tortuosity': 1.6}  # High tortuosity
    }
    
    print("\n[TEST 3] Hypertensive Retinopathy:")
    result3 = classify_diseases(htn_results, patient_age=58)
    print(f"Top Diagnosis: {result3['top_diagnosis']} ({result3['top_confidence']}%)")
    
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE")
    print("="*80)
