# RetinaGuard V500 — Calibration Analysis

> [!CAUTION]
> The calibration data reveals **fundamental problems** in the current system. Only **2 out of 10 experts** actually work. The other 8 are adding pure noise or are **inverted** (triggering MORE on healthy images than RP).

## Feature Discriminative Power (Ranked)

| Rank | Feature | J Score | RP Mean | Healthy Mean | Status |
|------|---------|---------|---------|--------------|--------|
| 1 | **AI Confidence** | **0.576** | 62.7% | 41.2% | ✅ WORKS — Best discriminator |
| 2 | **Vessel Density** | **0.419** | 0.070 | 0.121 | ✅ WORKS — RP has lower density |
| 3 | Quadrant Asymmetry | 0.200 | 0.072 | 0.022 | ⚠️ WEAK — Some signal |
| 4 | Hemorrhages | 0.179 | 2.73 | 1.00 | ⚠️ WEAK — May be bone spicules |
| 5 | Pigment Clusters | 0.142 | 1.53 | 0.06 | ❌ BROKEN — Threshold too high (MILD=8, but RP avg=1.5) |
| 6 | CME Score | 0.121 | 0.151 | 0.150 | ❌ USELESS — Identical for RP and Healthy |
| 7 | Bright Flecks | 0.065 | 9.3 | 1.3 | ⚠️ WEAK — Huge variance (std=42) |
| 8 | Local Variation | 0.050 | 2.64 | 3.01 | ❌ **INVERTED** — Healthy > RP |
| 9 | Microaneurysms | 0.042 | 29.4 | 50.8 | ❌ **INVERTED** — Healthy has MORE |
| 10 | Disc Brightness | 0.034 | 199.3 | 210.4 | ❌ **INVERTED** — Healthy is brighter |
| 11 | Texture Entropy | 0.025 | 6.28 | 6.53 | ❌ **INVERTED** — Healthy has more entropy |
| 12 | Spatial Degradation | 0.012 | 0.167 | 0.255 | ❌ **INVERTED** — Healthy shows MORE degradation |
| 13 | Vessel Tortuosity | 0.000 | 1.000 | 1.000 | ❌ **DEAD** — Returns 1.0 for everything |

## Root Cause Analysis

### Problem 1: Inverted Features (5 experts)
Disc Brightness, Texture Entropy, Local Variation, Spatial Degradation, and Microaneurysms all produce **higher values for Healthy images**. This means:
- The disc pallor detector triggers MORE on healthy eyes (bright disc = normal, not pallor)
- The texture/spatial experts add false RP votes to healthy images
- The microaneurysm detector is detecting **normal retinal features** as lesions

### Problem 2: Dead Features (2 experts)
- Tortuosity always returns 1.0 — the skeletonization is failing
- CME Score is identical for both classes — not discriminating at all

### Problem 3: Thresholds Set From Textbooks, Not Data
- Pigment MILD threshold = 8 clusters, but RP images average only 1.5 clusters
- Vessel MILD threshold = 0.25, but healthy images average 0.12 (most healthy images are already "attenuated" by this threshold!)

## The Fix Plan

### Step 1: Recalibrate Expert Weights
Give weight to features that actually work, zero out broken ones:

```
EXPERT_WEIGHTS (NEW — data-driven):
  ai_pattern_recognition: 0.35    (was 0.20 — this is the best feature)
  vessel_attenuation:     0.25    (was 0.16 — second best feature)  
  pigment_bone_spicules:  0.12    (was 0.14 — weak but real signal)
  quadrant:               0.08    (was 0.07 — weak signal)
  bright_lesion:          0.05    (was 0.08 — very weak)
  vessel_tortuosity:      0.05    (was 0.08 — dead, placeholder)
  optic_disc_pallor:      0.04    (was 0.10 — INVERTED, heavily reduced)
  texture_degeneration:   0.03    (was 0.06 — INVERTED, heavily reduced)
  spatial_pattern:        0.02    (was 0.04 — INVERTED, heavily reduced)
  macula:                 0.01    (was 0.07 — useless for RP detection)
```

### Step 2: Recalibrate Thresholds

```
AI Thresholds (NEW):
  AI_CRITICAL:           0.70    (keep — still valid)
  AI_MODERATE:           0.50    (was 0.55 — data says 0.50 is optimal)
  AI_MILD:               0.30    (was 0.25 — raise slightly)
  AI_POSITIVE_THRESHOLD: 0.50    (was 0.60 — data says 0.50)
  AI_UNCERTAIN_THRESHOLD: 0.40   (was 0.50 — widen uncertain zone)

Vessel Thresholds (NEW — based on actual density distributions):
  VESSEL_CRITICAL: 0.04    (was 0.08 — RP mean vessel=0.07)
  VESSEL_MODERATE: 0.07    (was 0.15 — optimal Youden threshold)
  VESSEL_MILD:     0.12    (was 0.25 — healthy mean=0.12)

Pigment Thresholds (NEW — based on RP avg=1.5 clusters):
  PIGMENT_CRITICAL: 8     (was 30 — almost never reached)
  PIGMENT_MODERATE: 3     (was 18 — almost never reached)
  PIGMENT_MILD:     1     (was 8 — RP average is 1.5!)
```

### Step 3: Fix the Decision Engine
- **Rule 0 (Differential Override)**: Already fixed — made stricter
- **Rule 5 (3+ clinical votes)**: Raise to 4+ votes since inverted experts inflate counts
- **Rule 6 (AI hallucination)**: Lower AI threshold from 0.60 to 0.50
- Deprioritize inverted experts from clinical vote counting

### Step 4: Fix Multi-Disease Classifier
- Microaneurysm normalization is wrong (healthy avg=50.8, divisor=30 makes every healthy image look like DR)
- Fix the `_extract_features` normalization divisors based on actual distributions
