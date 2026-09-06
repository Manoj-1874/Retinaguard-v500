# ==============================================================================
# FINAL ARCHITECTURE SOLUTIONS: HEURISTIC V13 ARTIFACT REJECTOR
# ==============================================================================
# Role in Research Paper:
# This script documents the immediate architectural consequence of the V12 "Ensemble Boost". 
# It proves that in heuristic programming, every "fix" inevitably creates a new failure.
#
# The Experiment:
# In V12, the researcher added a flat +0.30 confidence boost to any Color scan with 
# >3.0% pigment to save `rp_0.jpg`. 
# However, this instantly broke `norm_0.jpg` (a healthy 4-image grid). The grid's dark 
# borders scored 17.09% pigment. The V12 logic triggered the "🚀 BOOST", inflating 
# the AI's 0.31 score to 0.61. Because 0.61 > 0.50 threshold, the healthy image was 
# about to be diagnosed as a False Positive!
#
# To fix this, the researcher built V13, introducing an "ARTIFACT CHECK":
# If an image has massive pigment (>10.0%) but the AI is still hesitant (<80% confident), 
# assume it's a camera artifact and REJECT IT.
#
# Findings:
# The simulator perfectly caught `norm_0.jpg`. It boosted the score to 0.61, but 
# because 0.61 < 0.80, it hit the artifact gate and was rejected. The simulation 
# passed perfectly.
#
# Academic Conclusion:
# This is the textbook definition of technical debt in medical AI. The architecture 
# is now a stack of contradictory patches: "Boost the score to save this image, but 
# reject it if it gets boosted too high to save this other image." 
#
# Crucially, this sets the stage for the final V14 collapse. The V13 Artifact Check 
# relied on the AI being "hesitant" (<80% conf). But what happens when the AI is 
# 100% confident a healthy mosaic (`norm_1.jpg`) is sick? The V13 logic fails, 
# forcing the researcher to abandon pixels entirely and build the V14 "Geometry Bouncer".
# ==============================================================================

import numpy as np
import pandas as pd

# [V13 Artifact Rejector Implementation Truncated for Archive]
# Full source identical to Final V13 Simulator codebase.

# TERMINAL OUTPUT ARCHIVE:
# FILE         | PIGM   | RAW   | FINAL | RESULT     | STATUS | LOGIC PATH
# --------------------------------------------------------------------------------------------------------------
# rp_0.jpg     | 5.49   | 0.344 | 0.64  | RP         | ✅ PASS | 🚀 BOOST: +0.3 (Pigment 5.49% > 3%) -> 🎨 COLOR: Thresh is 0.5
# susp_0.jpg   | 2.5    | 0.69  | 0.69  | HEALTHY    | ✅ PASS | ⚠️ NOISY FAF: Thresh raised to 0.75
# norm_0.jpg   | 17.09  | 0.31  | 0.61  | HEALTHY    | ✅ PASS | 🚀 BOOST: +0.3 (Pigment 17.09% > 3%) -> 🛑 ARTIFACT: High Pigment (17.09%) but Low Conf (0.61). REJECTED.
# rp_10.jpg    | 39.35  | 0.871 | 0.87  | RP         | ✅ PASS | ⚠️ NOISY FAF: Thresh raised to 0.75
# --------------------------------------------------------------------------------------------------------------
