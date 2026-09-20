# Fix Plan for RetinaGuard V500 Metrics Achievement
# Target: Accuracy ≥94.9%, Precision ≥93.0%, Recall ≥97.1%, F1 ≥95.0%
# Current: Accuracy 61.96%, Precision 56.78%, Recall 97.81%, F1 71.85%
# Math: With 137 RP + 139 Healthy = 276 total
#   - Max FNs allowed: 4 (Recall 97.1% = 133/137)
#   - Max FPs allowed: ~10 (Precision 93% = 133/143, Accuracy 94.9% = 262/276)

# ROOT CAUSE ANALYSIS:
# 1. Sine Pigmento pathway: vessel_abnormal includes MILD severity 
#    → ANY image with vessel density <12% triggers it with AI >40%
#    → MANY healthy eyes have 8-12% vessel density (imaging artifacts)
#    FIX: Require MODERATE/CRITICAL vessels for Sine Pigmento, not MILD
#
# 2. Rule 5c: critical_count >= 1 and ai_confidence >= 0.35 
#    → Optic disc brightness >210 is very common in healthy images
#    → Threshold of 35% AI is absurdly low
#    FIX: Raise to ai_confidence >= 0.55 for isolated critical findings
#
# 3. Rule 5b-5k outer gate thresholds are too low
#    → mild_findings >= 1 and ai_confidence >= 0.50 triggers SUSPICIOUS
#    → mild_findings >= 2 and ai_confidence >= 0.45 triggers SUSPICIOUS
#    FIX: Raise all thresholds substantially
#
# 4. Optic disc DISC_MODERATE=195 is too low
#    → Many healthy fundus images show disc brightness 195-210
#    FIX: Raise DISC_MODERATE to 210 and DISC_MILD to 195
#
# 5. VESSEL_MILD=0.12 is too aggressive
#    → Healthy images routinely show 8-15% vessel density
#    FIX: Lower to 0.08 (8%) so only truly abnormal vessels fire MILD
#
# 6. Rule 3 (line 2020): ai_says_rp and (critical_count >= 1)
#    → A single critical optic disc + AI at 60% = RP_POSITIVE
#    → This is too easy to trigger
#    FIX: Require critical_count >= 2 OR (critical_count >= 1 AND has_pathognomonic)
