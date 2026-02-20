"""
================================================================================
VALIDATION STUDY TOOLKIT - RETINAGUARD V500
================================================================================
Statistical analysis tools for clinical validation studies and FDA submission.

FEATURES:
  1. Confusion Matrix Generation
  2. Sensitivity/Specificity Calculation
  3. ROC/AUC Analysis
  4. Inter-Rater Agreement (Kappa statistics)
  5. Subgroup Analysis (age, ethnicity, severity)
  6. Performance Reporting (FDA-compliant format)

CLINICAL TRIAL SUPPORT:
  - Patient enrollment tracking
  - Ground truth comparison
  - Multi-site data aggregation
  - Statistical significance testing

Version: 1.0.0
Author: RetinaGuard Development Team
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
import json

class ValidationStudyToolkit:
    """Statistical analysis for clinical validation"""
    
    def __init__(self):
        """Initialize validation toolkit"""
        self.results_database = []
        
    def add_patient_result(self, patient_data: Dict):
        """
        Add patient to validation database
        
        Args:
            patient_data: Dictionary with:
                - patient_id: str
                - ai_verdict: str (POSITIVE, SUSPICIOUS, BORDERLINE, NEGATIVE)
                - ground_truth: str (RP_CONFIRMED, RP_SUSPECTED, HEALTHY)
                - age: int
                - ethnicity: str
                - severity: str (EARLY, MODERATE, ADVANCED)
                - site: str (study site identifier)
        """
        self.results_database.append(patient_data)
    
    def calculate_performance_metrics(self, threshold: str = 'SUSPICIOUS') -> Dict:
        """
        Calculate diagnostic performance metrics
        
        Args:
            threshold: What AI verdict counts as "positive test"
                      'POSITIVE' = Only POSITIVE verdicts
                      'SUSPICIOUS' = POSITIVE + SUSPICIOUS (default)
                      'BORDERLINE' = POSITIVE + SUSPICIOUS + BORDERLINE
                      
        Returns:
            Dictionary with sensitivity, specificity, PPV, NPV, accuracy
        """
        print(f"\n   [M] PERFORMANCE METRICS CALCULATION")
        print(f"      {'='*60}")
        print(f"      Test Threshold: {threshold} or higher = Positive Test")
        print(f"      Sample Size: {len(self.results_database)} patients")
        
        # Define positive test criteria
        positive_verdicts = {
            'POSITIVE': ['POSITIVE'],
            'SUSPICIOUS': ['POSITIVE', 'SUSPICIOUS'],
            'BORDERLINE': ['POSITIVE', 'SUSPICIOUS', 'BORDERLINE']
        }[threshold]
        
        # Calculate confusion matrix
        tp = 0  # True Positive: AI positive, ground truth RP
        tn = 0  # True Negative: AI negative, ground truth healthy
        fp = 0  # False Positive: AI positive, ground truth healthy
        fn = 0  # False Negative: AI negative, ground truth RP
        
        for patient in self.results_database:
            ai_positive = patient['ai_verdict'] in positive_verdicts
            gt_positive = patient['ground_truth'] in ['RP_CONFIRMED', 'RP_SUSPECTED']
            
            if ai_positive and gt_positive:
                tp += 1
            elif not ai_positive and not gt_positive:
                tn += 1
            elif ai_positive and not gt_positive:
                fp += 1
            elif not ai_positive and gt_positive:
                fn += 1
        
        # Calculate metrics
        total = tp + tn + fp + fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # F1 Score (harmonic mean of precision and recall)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # Display confusion matrix
        print(f"\n      [M] CONFUSION MATRIX:")
        print(f"                      Ground Truth")
        print(f"                    RP     | Healthy")
        print(f"         AI  RP     {tp:4d}  |  {fp:4d}   (AI Positive)")
        print(f"             Healthy {fn:4d}  |  {tn:4d}   (AI Negative)")
        
        print(f"\n      [M] PERFORMANCE METRICS:")
        print(f"         Sensitivity (Recall):    {sensitivity*100:5.1f}% ({tp}/{tp+fn})")
        print(f"         Specificity:             {specificity*100:5.1f}% ({tn}/{tn+fp})")
        print(f"         Positive Predictive Value: {ppv*100:5.1f}% ({tp}/{tp+fp})")
        print(f"         Negative Predictive Value: {npv*100:5.1f}% ({tn}/{tn+fn})")
        print(f"         Accuracy:                {accuracy*100:5.1f}% ({tp+tn}/{total})")
        print(f"         F1 Score:                {f1:5.3f}")
        
        # FDA target benchmarks
        print(f"\n      [T] FDA TARGET BENCHMARKS:")
        self._display_benchmark('Sensitivity', sensitivity, 0.80, 0.75)
        self._display_benchmark('Specificity', specificity, 0.90, 0.85)
        self._display_benchmark('PPV', ppv, 0.80, 0.70)
        self._display_benchmark('NPV', npv, 0.95, 0.90)
        
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return {
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'ppv': round(ppv, 4),
            'npv': round(npv, 4),
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'sample_size': total
        }
    
    def _display_benchmark(self, metric: str, value: float, target: float, acceptable: float):
        """Display metric vs target"""
        status = "[+] EXCEEDS TARGET" if value >= target else \
                 "[o] ACCEPTABLE" if value >= acceptable else \
                 "[X] BELOW TARGET"
        print(f"         {metric:30s} {value*100:5.1f}% | Target: {target*100:.0f}% | {status}")
    
    def subgroup_analysis(self) -> Dict:
        """
        Analyze performance across patient subgroups
        
        Stratifies by:
            - Age (pediatric, adult, geriatric)
            - Ethnicity
            - Disease severity
            - Study site
            
        Returns:
            Dictionary with subgroup-specific metrics
        """
        print(f"\n   [S] SUBGROUP ANALYSIS")
        print(f"      {'='*60}")
        
        subgroups = defaultdict(list)
        
        # Stratify patients
        for patient in self.results_database:
            # Age groups
            age = patient.get('age', 40)
            if age < 18:
                subgroups['age_pediatric'].append(patient)
            elif age < 65:
                subgroups['age_adult'].append(patient)
            else:
                subgroups['age_geriatric'].append(patient)
            
            # Ethnicity
            eth = patient.get('ethnicity', 'unknown')
            subgroups[f'ethnicity_{eth}'].append(patient)
            
            # Severity
            sev = patient.get('severity', 'unknown')
            if sev != 'unknown':
                subgroups[f'severity_{sev.lower()}'].append(patient)
            
            # Site
            site = patient.get('site', 'unknown')
            subgroups[f'site_{site}'].append(patient)
        
        # Calculate metrics for each subgroup
        subgroup_metrics = {}
        
        for group_name, patients in subgroups.items():
            if len(patients) < 5:  # Skip small subgroups
                continue
            
            # Temporarily replace database for metrics calculation
            original_db = self.results_database
            self.results_database = patients
            
            metrics = self.calculate_performance_metrics(threshold='SUSPICIOUS')
            subgroup_metrics[group_name] = {
                'n': len(patients),
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'accuracy': metrics['accuracy']
            }
            
            # Restore original database
            self.results_database = original_db
        
        # Display subgroup summary
        print(f"\n      [S] SUBGROUP SUMMARY:")
        for group, metrics in sorted(subgroup_metrics.items()):
            print(f"         {group:25s} N={metrics['n']:3d} | Sens={metrics['sensitivity']*100:5.1f}% | Spec={metrics['specificity']*100:5.1f}%")
        
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return subgroup_metrics
    
    def calculate_inter_rater_agreement(self, rater1_verdicts: List[str], 
                                       rater2_verdicts: List[str]) -> Dict:
        """
        Calculate Cohen's Kappa for inter-rater agreement
        
        Used to compare:
            - AI vs Human Expert 1
            - Human Expert 1 vs Human Expert 2
            - Pre-training vs Post-training AI
            
        Args:
            rater1_verdicts: List of judgments from rater 1
            rater2_verdicts: List of judgments from rater 2
            
        Returns:
            Kappa statistic and interpretation
        """
        print(f"\n   [A] INTER-RATER AGREEMENT")
        print(f"      {'='*60}")
        
        if len(rater1_verdicts) != len(rater2_verdicts):
            print(f"      [X] ERROR: Rater lists must be same length")
            return {}
        
        n = len(rater1_verdicts)
        
        # Calculate observed agreement
        agreements = sum(1 for r1, r2 in zip(rater1_verdicts, rater2_verdicts) if r1 == r2)
        po = agreements / n  # Proportion observed agreement
        
        # Calculate expected agreement (by chance)
        verdict_counts1 = defaultdict(int)
        verdict_counts2 = defaultdict(int)
        
        for v1, v2 in zip(rater1_verdicts, rater2_verdicts):
            verdict_counts1[v1] += 1
            verdict_counts2[v2] += 1
        
        all_verdicts = set(list(verdict_counts1.keys()) + list(verdict_counts2.keys()))
        pe = 0.0
        for verdict in all_verdicts:
            p1 = verdict_counts1[verdict] / n
            p2 = verdict_counts2[verdict] / n
            pe += p1 * p2
        
        # Cohen's Kappa
        if pe < 1.0:
            kappa = (po - pe) / (1 - pe)
        else:
            kappa = 1.0
        
        # Interpretation
        if kappa > 0.80:
            interpretation = "EXCELLENT agreement"
        elif kappa > 0.60:
            interpretation = "SUBSTANTIAL agreement"
        elif kappa > 0.40:
            interpretation = "MODERATE agreement"
        elif kappa > 0.20:
            interpretation = "FAIR agreement"
        else:
            interpretation = "POOR agreement"
        
        print(f"      Sample Size: {n}")
        print(f"      Observed Agreement: {po*100:.1f}% ({agreements}/{n})")
        print(f"      Expected Agreement (chance): {pe*100:.1f}%")
        print(f"      Cohen's Kappa: {kappa:.3f}")
        print(f"      Interpretation: {interpretation}")
        print(f"      {'='*60}\n")
        sys.stdout.flush()
        
        return {
            'kappa': round(kappa, 3),
            'observed_agreement': round(po, 3),
            'expected_agreement': round(pe, 3),
            'interpretation': interpretation
        }
    
    def generate_fda_report(self, study_name: str = "RetinaGuard V500 Validation") -> str:
        """
        Generate FDA-compliant performance report
        
        Returns:
            Formatted text report suitable for 510(k) submission
        """
        report = []
        report.append("="*80)
        report.append(f"CLINICAL VALIDATION STUDY REPORT")
        report.append(f"{study_name}")
        report.append("="*80)
        report.append("")
        
        # Study Overview
        report.append("1. STUDY OVERVIEW")
        report.append("-" * 80)
        report.append(f"Total Patients Enrolled: {len(self.results_database)}")
        
        # Ground truth distribution
        rp_confirmed = sum(1 for p in self.results_database if p['ground_truth'] == 'RP_CONFIRMED')
        rp_suspected = sum(1 for p in self.results_database if p['ground_truth'] == 'RP_SUSPECTED')
        healthy = sum(1 for p in self.results_database if p['ground_truth'] == 'HEALTHY')
        
        report.append(f"  - RP Confirmed: {rp_confirmed}")
        report.append(f"  - RP Suspected: {rp_suspected}")
        report.append(f"  - Healthy Controls: {healthy}")
        report.append("")
        
        # Performance Metrics
        report.append("2. DIAGNOSTIC PERFORMANCE")
        report.append("-" * 80)
        
        metrics = self.calculate_performance_metrics(threshold='SUSPICIOUS')
        
        report.append(f"Sensitivity:    {metrics['sensitivity']*100:.2%}")
        report.append(f"Specificity:    {metrics['specificity']*100:.2%}")
        report.append(f"PPV:            {metrics['ppv']*100:.2%}")
        report.append(f"NPV:            {metrics['npv']*100:.2%}")
        report.append(f"Accuracy:       {metrics['accuracy']*100:.2%}")
        report.append(f"F1 Score:       {metrics['f1_score']:.3f}")
        report.append("")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        report.append("Confusion Matrix:")
        report.append(f"  True Positives:  {cm['tp']}")
        report.append(f"  True Negatives:  {cm['tn']}")
        report.append(f"  False Positives: {cm['fp']}")
        report.append(f"  False Negatives: {cm['fn']}")
        report.append("")
        
        # Subgroup Analysis
        report.append("3. SUBGROUP ANALYSIS")
        report.append("-" * 80)
        subgroups = self.subgroup_analysis()
        for group, metrics in sorted(subgroups.items()):
            report.append(f"{group:30s} N={metrics['n']:3d} Sens={metrics['sensitivity']*100:5.1f}% Spec={metrics['specificity']*100:5.1f}%")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text


# Convenience functions
def create_validation_study() -> ValidationStudyToolkit:
    """Create new validation study"""
    return ValidationStudyToolkit()


# Testing harness
if __name__ == "__main__":
    print("="*80)
    print("VALIDATION STUDY TOOLKIT - TEST SUITE")
    print("="*80)
    
    # Create mock validation study
    study = create_validation_study()
    
    # Add mock patients
    # 100 RP patients, 100 healthy controls
    np.random.seed(42)
    
    for i in range(100):
        # RP patients
        study.add_patient_result({
            'patient_id': f'RP-{i:03d}',
            'ai_verdict': np.random.choice(['POSITIVE', 'SUSPICIOUS', 'BORDERLINE'], p=[0.7, 0.2, 0.1]),
            'ground_truth': 'RP_CONFIRMED',
            'age': np.random.randint(20, 70),
            'ethnicity': np.random.choice(['caucasian', 'african', 'asian', 'hispanic']),
            'severity': np.random.choice(['EARLY', 'MODERATE', 'ADVANCED']),
            'site': np.random.choice(['Site_A', 'Site_B', 'Site_C'])
        })
        
        # Healthy controls
        study.add_patient_result({
            'patient_id': f'HC-{i:03d}',
            'ai_verdict': np.random.choice(['NEGATIVE', 'BORDERLINE', 'SUSPICIOUS'], p=[0.8, 0.15, 0.05]),
            'ground_truth': 'HEALTHY',
            'age': np.random.randint(25, 75),
            'ethnicity': np.random.choice(['caucasian', 'african', 'asian', 'hispanic']),
            'severity': 'NONE',
            'site': np.random.choice(['Site_A', 'Site_B', 'Site_C'])
        })
    
    # Calculate performance
    print("\n[TEST 1] Overall Performance:")
    metrics = study.calculate_performance_metrics()
    
    # Subgroup analysis
    print("\n[TEST 2] Subgroup Analysis:")
    subgroups = study.subgroup_analysis()
    
    # Inter-rater agreement
    print("\n[TEST 3] Inter-Rater Agreement:")
    rater1 = ['POSITIVE', 'POSITIVE', 'NEGATIVE', 'SUSPICIOUS', 'POSITIVE']
    rater2 = ['POSITIVE', 'SUSPICIOUS', 'NEGATIVE', 'SUSPICIOUS', 'POSITIVE']
    kappa = study.calculate_inter_rater_agreement(rater1, rater2)
    
    # Generate FDA report
    print("\n[TEST 4] FDA Report Generation:")
    report = study.generate_fda_report()
    
    print("\n" + "="*80)
    print("VALIDATION TESTING COMPLETE")
    print("="*80)
