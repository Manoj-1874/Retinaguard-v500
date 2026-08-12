# Domain Research: Healthcare Informatics & Retinal Image Processing

## 1. The Domain
The project operates within **Medical Artificial Intelligence (Healthcare Informatics)**, specifically focusing on **Ophthalmology and Retinal Image Processing**.

## 2. Background
Retinal imaging (Fundus Photography) is the primary non-invasive method for observing the microvascular network of the human body. Traditional diagnosis requires a highly trained ophthalmologist to manually inspect the fundus image for microscopic anomalies (e.g., bone spicule pigment, macular edema, vessel attenuation). 

Due to the global shortage of trained specialists, particularly in rural and low-resource areas, Artificial Intelligence has emerged as a vital tool for automated screening.

## 3. Retinitis Pigmentosa (RP)
RP is a group of rare genetic disorders that involve a breakdown and loss of cells in the retina. The clinical diagnosis of classic RP relies on identifying the "Classic Triad":
1. **Bone Spicule Pigmentation:** Dark, melanin-based deposits in the mid-periphery.
2. **Vessel Attenuation:** Severe thinning of the retinal blood vessels.
3. **Optic Disc Pallor:** A waxy, pale yellowing of the optic nerve.

## 4. The Challenge of "Variant" Diseases
While classic RP is well-documented, the domain suffers from immense complexity due to variants:
- **Retinitis Punctata Albescens (RPA):** Features bright white flecks instead of dark pigment.
- **Sine Pigmento:** Severe RP progression with absolutely zero visible pigment.
- **Sectoral RP:** Asymmetrical presentation limited to a single quadrant of the eye.

Standard Deep Learning classifiers fail to handle these edge cases because they rely on generalized pattern recognition rather than strict medical logic. Therefore, the domain is shifting towards **Clinical Decision Support Systems (CDSS)** which combine AI pattern recognition with hardcoded, transparent medical rules.
