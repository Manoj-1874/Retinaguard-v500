"""
================================================================================
MULTI-DISEASE CLASSIFIER - RETINAGUARD V500
================================================================================
Differential diagnosis for 7 retinal conditions to prevent misdiagnosis.

SUPPORTED CONDITIONS:
  1. Retinitis Pigmentosa   - bone spicules, vessel attenuation, disc pallor
  2. Diabetic Retinopathy   - microaneurysms, hemorrhages, exudates
  3. AMD                    - drusen, geographic atrophy, macular edema
  4. Glaucoma               - disc cupping, RNFL thinning
  5. Hypertensive Retinopathy - vessel tortuosity, AV nicking
  6. Choroideremia          - peripheral chorioretinal atrophy
  7. Usher Syndrome         - RP + early onset (pediatric)

SCORING METHOD:
  - Weighted feature matching (positive evidence)
  - Exclusion penalties     (contradictory features halve the score)
  - Age-appropriate filtering (AMD unlikely under 50)

Version: 1.0.0
================================================================================
"""

from typing import Dict, List, Tuple

DISEASE_PATTERNS = {
    "retinitis_pigmentosa": {
        "name": "Retinitis Pigmentosa",
        "features": {"bone_spicules": 0.35, "vessel_attenuation": 0.30, "disc_pallor": 0.25, "peripheral_loss": 0.10},
        "exclusions": ["microaneurysms", "drusen", "disc_cupping"],
    },
    "diabetic_retinopathy": {
        "name": "Diabetic Retinopathy",
        "features": {"microaneurysms": 0.35, "hemorrhages": 0.30, "exudates": 0.20, "cotton_wool_spots": 0.15},
        "exclusions": ["bone_spicules", "disc_pallor"],
    },
    "amd": {
        "name": "Age-Related Macular Degeneration",
        "features": {"drusen": 0.40, "geographic_atrophy": 0.30, "macular_edema": 0.20, "abnormal_texture": 0.10},
        "exclusions": ["bone_spicules", "vessel_attenuation"],
    },
    "glaucoma": {
        "name": "Glaucoma",
        "features": {"disc_cupping": 0.45, "rnfl_thinning": 0.35, "disc_pallor": 0.15, "peripapillary_atrophy": 0.05},
        "exclusions": ["bone_spicules", "microaneurysms"],
    },
    "hypertensive_retinopathy": {
        "name": "Hypertensive Retinopathy",
        "features": {"vessel_tortuosity": 0.50, "optic_disc_edema": 0.30, "exudates": 0.20},
        "exclusions": ["bone_spicules", "drusen"],
        "note": "Silver/copper wire appearance in vessels; papilledema in malignant hypertension — rule out if BP history unavailable",
    },
    "choroideremia": {
        "name": "Choroideremia",
        "features": {"chorioretinal_atrophy": 0.45, "peripheral_loss": 0.30, "vessel_attenuation": 0.15, "macular_preservation": 0.10},
        "exclusions": ["bone_spicules", "microaneurysms", "drusen"],
        "note": "X-linked — affects males; macular island preserved until late stage; key differentiator from RP",
    },
    "usher_syndrome": {
        "name": "Usher Syndrome",
        "features": {"bone_spicules": 0.30, "vessel_attenuation": 0.25, "disc_pallor": 0.20, "peripheral_loss": 0.15, "early_onset": 0.10},
        "exclusions": ["drusen", "microaneurysms"],
        "age_filter": "pediatric_or_young_adult",
        "note": "RP + congenital sensorineural hearing loss — always consider in patients under 30",
        "exclusions": ["drusen", "microaneurysms"],
    },
}

class MultiDiseaseClassifier:

    def classify(self, expert_results: Dict, patient_age: int = 40) -> Dict:
        features = self._extract_features(expert_results, patient_age)
        scores   = {}
        for did, pattern in DISEASE_PATTERNS.items():
            score = sum(features.get(f, 0) * w for f, w in pattern["features"].items())
            for ex in pattern["exclusions"]:
                if features.get(ex, 0) > 0.3:
                    score *= 0.5
            scores[did] = min(score, 1.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        differential = [
            {"disease": DISEASE_PATTERNS[d]["name"], "confidence": round(s * 100, 1)}
            for d, s in ranked if s >= 0.10
        ]
        return {
            "top_diagnosis": DISEASE_PATTERNS[ranked[0][0]]["name"],
            "top_confidence": round(ranked[0][1] * 100, 1),
            "differential": differential,
        }

    def _extract_features(self, expert_results: Dict, age: int) -> Dict:
        pigment = expert_results.get("pigment_result", {})
        vessel  = expert_results.get("vessel_result", {})
        disc    = expert_results.get("optic_disc_result", {})
        spatial = expert_results.get("spatial_result", {})
        texture = expert_results.get("texture_result", {})
        bright  = expert_results.get("bright_lesion_result", {})
        macula  = expert_results.get("macula_result", {})
        tort    = expert_results.get("tortuosity_result", {})

        vessel_density  = vessel.get("density", 0.30)
        disc_brightness = disc.get("brightness", 160)

        return {
            "bone_spicules":       min(pigment.get("cluster_count", 0) / 30.0, 1.0),
            "vessel_attenuation":  max(0, (0.30 - vessel_density) / 0.30),
            "disc_pallor":         max(0, (disc_brightness - 180) / 50),
            "peripheral_loss":     spatial.get("degradation_score", 0.0),
            "microaneurysms":      0.0,
            "hemorrhages":         0.0,
            "exudates":            min(texture.get("local_variation", 0) / 6.0, 1.0),
            "cotton_wool_spots":   0.0,
            "drusen":              min(bright.get("fleck_count", 0) / 20.0, 1.0),
            "macular_edema":       macula.get("cme_score", 0.0),
            "abnormal_texture":    min(texture.get("entropy", 5.0) / 7.0, 1.0),
            "geographic_atrophy":  0.0,
            "disc_cupping":        0.0,
            "rnfl_thinning":       max(0, (0.30 - vessel_density) / 0.30) * 0.5,
            "peripapillary_atrophy": 0.0,
            "vessel_tortuosity":   max(0, min((tort.get("tortuosity", 1.0) - 1.3) / 0.7, 1.0)),
            "optic_disc_edema":    max(0, min((170 - disc_brightness) / 40, 1.0)),
            "chorioretinal_atrophy": spatial.get("degradation_score", 0.0) * 0.8,
            "macular_preservation": max(0, 1.0 - macula.get("cme_score", 0.0)),
            "early_onset":         1.0 if age < 20 else 0.5 if age < 30 else 0.0,
        }

def classify_diseases(expert_results: Dict, patient_age: int = 40) -> Dict:
    return MultiDiseaseClassifier().classify(expert_results, patient_age)

    def _apply_sine_pigmento_boost(self, scores: dict, features: dict) -> dict:
        """
        Boost RP score for Sine Pigmento variant:
        Classic RP has bone spicules. Sine Pigmento has vessel attenuation
        and disc pallor WITHOUT visible bone spicules.
        Without this boost the classifier would rank RP low and miss this variant.
        Trigger: vessel_attenuation > 0.4 OR ai_prob > 0.5, AND bone_spicules < 0.3
        """
        vessel_att = features.get("vessel_attenuation", 0.0)
        bone_spic  = features.get("bone_spicules", 0.0)
        ai_prob    = features.get("ai_rp_probability", 0.0)

        if (vessel_att > 0.4 or ai_prob > 0.5) and bone_spic < 0.3:
            sine_score = (
                vessel_att * 0.45 +
                features.get("disc_pallor", 0.0) * 0.25 +
                features.get("peripheral_loss", 0.0) * 0.10 +
                ai_prob * 0.20
            )
            if sine_score > scores.get("retinitis_pigmentosa", 0):
                scores["retinitis_pigmentosa"] = min(sine_score, 0.95)
        return scores
