# Problem Statement: AI in Retinal Diagnostics

## 1. Introduction
Retinitis Pigmentosa (RP) is a severe, progressive genetic disorder that leads to vision loss and eventual blindness. Early and accurate diagnosis is critical for patient management. While artificial intelligence (AI), particularly Convolutional Neural Networks (CNNs), has shown promise in automating retinal disease detection, current models suffer from severe limitations that prevent clinical deployment.

## 2. Core Limitations of Existing Systems
1. **The "Black Box" Problem:** Deep learning models output probability scores without providing clinical rationale. In a medical setting, an unexplainable diagnosis is legally and ethically unacceptable.
2. **High False Positive Rates:** Basic AI models cannot differentiate between visually similar pathologies. For example, the dark red hemorrhages seen in Diabetic Retinopathy are frequently misclassified as the dark melanin "bone spicules" seen in RP.
3. **Hardware Vulnerability:** Most medical AI is trained on pristine datasets captured by $50,000 tabletop fundus cameras. When exposed to real-world, low-resource environments using affordable handheld cameras, the resulting blurry, underexposed, or color-shifted images cause catastrophic AI failure.
4. **Variant Blindness:** AI models are rigid and struggle to identify rare sub-variants of diseases, such as Sine Pigmento (RP without pigment) or Retinitis Punctata Albescens (RPA).

## 3. Proposed Solution
This project introduces **RetinaGuard V500**, an advanced Clinical Decision Support System (CDSS) that replaces the vulnerable "Black Box" with a 10-Expert Rule-Based Engine. By extracting mathematically provable biological features (vessel density, pigment clusters, optic disc pallor) and utilizing dynamic color-space calibration, this system aims to achieve FDA-grade reliability across diverse hardware environments while providing fully transparent diagnostic reasoning.
