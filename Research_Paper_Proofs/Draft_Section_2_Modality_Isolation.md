# Section 2: The Failure of Heuristic Modality Isolation

## 2.1 The Multi-Modal Challenge in Retinal Imaging
A robust Clinical Decision Support System (CDSS) for Retinitis Pigmentosa must successfully parse multiple imaging modalities, primarily Color Fundus Photography (Color) and Fundus Autofluorescence (FAF). These modalities represent the exact same biological structures but render them with inverse pixel intensities. In a Color scan, pathological bone-spicules appear as dark pixel clusters against a bright orange retina. Conversely, in an FAF scan, the healthy macula appears as a large, dark void, while pathological autofluorescence may present as hyper-reflective (bright) rings or patches. 

Early attempts to automate biomarker extraction during the Colab Era relied on global pixel-density calculations (e.g., OpenCV's `cv2.threshold`). The underlying hypothesis was that pathological pigment could be isolated by establishing a universal "darkness" limit.

## 2.2 Proof 1: The Color Pigment Vulnerability
The initial heuristic engine applied a strict mathematical floor (`PIGMENT_HARD_LIMIT = 190` in the inverted blue channel) to segment bone-spicules in Color scans. When applied to standard, well-lit Color fundus images (e.g., `rp_0.jpg`), the heuristic successfully isolated pathological pigment clusters. 

However, this reliance on global pixel density proved extraordinarily brittle when exposed to natural physiological variance. When the system encountered a Tigroid fundus—a healthy eye with a naturally heavily-pigmented choroid—the engine catastrophically failed. Because a Tigroid fundus is globally darker than a standard eye, the static `cv2.threshold` operation indiscriminately captured the healthy choroidal vessels, inflating the pigment score to 60.18%. This forced the AI model to output an 82.5% False Positive, effectively misdiagnosing a perfectly healthy biological variant as severe, late-stage Retinitis Pigmentosa.

This failure demonstrated the first critical vulnerability of heuristic architecture: **Global pixel thresholds cannot differentiate between pathological structures and natural physiological variance.**

## 2.3 Proof 2: The Hardcoded FAF Fallacy
The fatal flaw of pixel-counting was further exposed when the heuristic engine was applied to Grayscale FAF scans. Because the engine was calibrated to look for "dark" pigment (to find bone-spicules in Color scans), it fundamentally misunderstood the topology of an FAF image. 

In a healthy FAF scan, the macula naturally absorbs light, appearing as a massive, dark circular region in the center of the retina. When the Colab engine processed a healthy FAF scan, the `cv2.threshold` operation blindly segmented this healthy macula, counting the massive dark void as a giant cluster of "disease." The system hallucinated a staggering 87.05% pigment score on a perfectly healthy eye, generating a 99.8% False Positive prediction.

To bandage this failure, researchers attempted to implement modality isolation—a bifurcated logic gate that forced the engine to apply completely different mathematical rules if the image was determined to be Grayscale. However, as demonstrated by the V28 and V40 architectures, "detecting" a modality via pixel math is just as brittle as detecting the disease itself. Noisy FAF scans, color scans with severe shadows, and multi-modal triptychs constantly bypassed these gates, routing images to the wrong mathematical logic and triggering cascading algorithmic collapse.

## 2.4 Conclusion: The Necessity of Topology
The cascading failures of the heuristic modality gates mathematically proved that a CDSS cannot treat an eye scan as a mere collection of dark and light pixels. It must understand *what* those pixels represent. A dark patch in the periphery of a Color scan is pathological; a dark patch in the center of an FAF scan is healthy. 

This spatial and modal context cannot be achieved through `cv2.threshold` or hardcoded Python `if` statements. It requires a fundamental paradigm shift towards topology-aware feature extraction, necessitating the abandonment of the Colab codebase and the transition to the Geometric Fragmentation Index.
