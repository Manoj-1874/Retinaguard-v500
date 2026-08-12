# RetinaGuard V500: System Architecture & Overcoming Limitations

## 1. System Overview
RetinaGuard V500 completely abandons the traditional "Black Box" AI approach. Instead, it utilizes a hybrid architecture that combines a base Deep Learning anomaly detector with a deterministic, mathematically verifiable **10-Expert Clinical Decision Support System (CDSS)**.

## 2. The 10-Expert Engine
The system analyzes every retinal image using 10 highly specialized Computer Vision algorithms (Experts) designed using OpenCV and Morphological mathematics:

1. **Vessel Attenuation Expert:** Uses adaptive Frangi filtering to calculate exact blood vessel density (%).
2. **Pigment Expert:** Uses LAB color-space isolation to detect pure melanin bone spicules.
3. **Optic Disc Expert:** Utilizes dynamic thresholding with strict area constraints (rejecting massive camera flashes).
4. **Tortuosity Expert:** Calculates vessel curvature to identify hypertensive or diabetic complications.
5. **Texture Degeneration Expert:** Measures global entropy to detect subtle tissue atrophy.
6. **Spatial Pattern Expert:** Compares peripheral vs. central degradation.
7. **RPA Lesion Expert:** Isolates high-intensity bright flecks.
8. **Macular Edema Expert:** Uses contour mapping to detect fluid buildup (CME).
9. **Sectoral Asymmetry Expert:** Divides the retina into quadrants to detect asymmetrical diseases.
10. **AI Neural Network:** Acts as a sensitive, but distrusted, initial anomaly detector.

## 3. How We Overcome Base Paper Limitations

### A. Eliminating False Positives (The Blood vs. Pigment Problem)
**Limitation:** Basic AI misdiagnoses dark red Diabetic Hemorrhages as RP Bone Spicules.
**Our Solution:** We implemented an **RGB/LAB Color-Space Filter**. The Pigment Expert mathematically subtracts pixels where the Red channel is significantly higher than Blue/Green. This explicitly filters out red blood, leaving only true black melanin pigment.

### B. Handling Hardware Variance (The Handheld Camera Problem)
**Limitation:** Models trained on Zeiss tabletop scanners reject or misdiagnose blurry, dark images from affordable handheld cameras.
**Our Solution:** We implemented **Dynamic Camera Calibration**. Before analysis, if a handheld camera is selected, the system automatically runs Gamma correction, CLAHE (Contrast Limited Adaptive Histogram Equalization), and brightness offsets. Furthermore, it dynamically relaxes the FDA "strict_mode" quality thresholds to allow the analysis to proceed.

### C. Variant Blindness
**Limitation:** AI models fail to detect rare RP variants.
**Our Solution:** We built specific **Deterministic Logical Pathways** into the Decision Engine:
- **Sine Pigmento:** Triggers if Vessels and Optic Disc are critical, but Pigment is absolutely 0.
- **Retinitis Punctata Albescens:** Triggers if Bright Flecks are detected alongside peripheral degeneration, with no dark pigment.

### D. Graceful Degradation
If the AI detects an anomaly (e.g., 75% confidence) but the 9 Clinical Experts measure the retina as biologically healthy, the Decision Engine **vetoes the AI**. Instead of generating a False Positive, the system safely degrades to a "Borderline / Monitor" verdict.
