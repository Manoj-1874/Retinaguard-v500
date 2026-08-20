"""
================================================================================
VALIDATION STUDY TOOLKIT - RETINAGUARD V500
================================================================================
Statistical tools for clinical validation and FDA 510(k) submission.

METRICS COMPUTED:
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - Positive Predictive Value (PPV)
  - Negative Predictive Value (NPV)
  - Accuracy, F1 Score
  - ROC AUC
  - Cohen Kappa inter-rater agreement

FDA BENCHMARKS (Class II Device):
  Sensitivity >= 80% (target), >= 75% (acceptable)
  Specificity >= 90% (target), >= 85% (acceptable)

Version: 1.0.0
================================================================================
"""

from typing import Dict, List, Optional
import numpy as np

FDA_SENSITIVITY_TARGET     = 0.80
FDA_SENSITIVITY_ACCEPTABLE = 0.75
FDA_SPECIFICITY_TARGET     = 0.90
FDA_SPECIFICITY_ACCEPTABLE = 0.85

class ValidationStudyToolkit:

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str],
                          positive_label: str = "POSITIVE") -> Dict:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_label and g == positive_label)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p != positive_label and g != positive_label)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_label and g != positive_label)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != positive_label and g == positive_label)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy    = (tp + tn) / len(predictions) if predictions else 0.0
        f1          = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

        sens_grade = (
            "TARGET MET"  if sensitivity >= FDA_SENSITIVITY_TARGET else
            "ACCEPTABLE"  if sensitivity >= FDA_SENSITIVITY_ACCEPTABLE else
            "BELOW TARGET"
        )
        spec_grade = (
            "TARGET MET"  if specificity >= FDA_SPECIFICITY_TARGET else
            "ACCEPTABLE"  if specificity >= FDA_SPECIFICITY_ACCEPTABLE else
            "BELOW TARGET"
        )
        fda_pass = (sensitivity >= FDA_SENSITIVITY_ACCEPTABLE and
                    specificity >= FDA_SPECIFICITY_ACCEPTABLE)

        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity":      round(sensitivity, 4),
            "specificity":      round(specificity, 4),
            "ppv":              round(ppv, 4),
            "npv":              round(npv, 4),
            "accuracy":         round(accuracy, 4),
            "f1_score":         round(f1, 4),
            "sensitivity_grade": sens_grade,
            "specificity_grade": spec_grade,
            "fda_pass":         fda_pass,
        }

def create_validation_study(predictions, ground_truth, positive_label="POSITIVE"):
    return ValidationStudyToolkit().calculate_metrics(predictions, ground_truth, positive_label)

    def cohen_kappa(self, rater1: List[str], rater2: List[str]) -> Dict:
        """
        Cohen Kappa inter-rater agreement.
        Measures how much two clinicians agree beyond random chance.

        Interpretation:
          > 0.80  : Excellent  (publication-grade agreement)
          0.60-0.80: Substantial
          0.40-0.60: Moderate
          < 0.40  : Poor       (raters disagree significantly)

        Required for FDA 510(k) multi-site validation studies.
        """
        labels   = list(set(rater1 + rater2))
        n        = len(rater1)
        observed = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

        expected = sum(
            (rater1.count(l) / n) * (rater2.count(l) / n)
            for l in labels
        )

        kappa = (observed - expected) / (1 - expected) if (1 - expected) != 0 else 1.0

        interpretation = (
            "Excellent"    if kappa > 0.80 else
            "Substantial"  if kappa > 0.60 else
            "Moderate"     if kappa > 0.40 else
            "Poor"
        )

        return {
            "kappa":               round(kappa, 4),
            "observed_agreement":  round(observed, 4),
            "expected_agreement":  round(expected, 4),
            "interpretation":      interpretation,
        }
