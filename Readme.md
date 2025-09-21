# Facial Geometric Analysis Tool

A computational tool for analyzing facial proportions using mathematical ratios and geometric principles, with emphasis on classical proportion theories including the golden ratio (φ ≈ 1.618).

## Overview

This tool extracts facial landmarks using MediaPipe Face Mesh and calculates various geometric relationships between facial features. It provides quantitative analysis of proportional relationships that have been studied in classical aesthetics and modern facial analysis research.

**Important Note**: This is a mathematical analysis tool for educational and research purposes. Facial proportions are highly variable across populations and cultures, and mathematical ratios do not define beauty or attractiveness.

## Technical Implementation

### Landmark Detection
- Uses MediaPipe Face Mesh with 468 3D facial landmarks
- Extracts key anatomical points for measurement
- Processes static images with single face detection

### Key Measurement Points
```
Vertical Structure:
- Forehead top (estimated from landmark 10)
- Eyebrow midpoint (landmarks 8, 9)
- Nose bridge (landmarks 168, 8)
- Nose tip (landmark 1)
- Chin (landmark 152)
...

Horizontal Structure:
- Left/Right cheek boundaries (landmarks 234, 454)
- Eye corners (landmarks 33, 133, 362, 263)
- Nose width (landmarks 48, 278)
- Mouth corners (landmarks 61, 291)
...
```

## Analyzed Ratios and Mathematical Basis

### 1. Golden Ratio Relationships (φ = 1.618)

**Face Length / Face Width**
- Classical proportion theory suggests ideal facial proportions follow φ
- Measured from forehead top to chin vs. cheek-to-cheek width

**Mouth Width / Nose Width**
- Based on Renaissance art proportional studies
- Theoretical ideal: φ ratio between these features

**Interocular Distance / Nose Width**
- Relationship between eye spacing and nose proportions
- φ ratio suggested in some aesthetic proportion theories

**Lower Lip / Upper Lip**
- Vertical lip proportion analysis
- φ ratio application to lip segment relationships

### 2. Fractional Proportions

**Mouth Width / Face Width = 0.38**
- Derived from φ relationships (1/φ ≈ 0.618, modified for facial analysis)
- Research suggests mouth width is approximately 38% of face width in some populations

**Forehead Length / Face Length = 0.333**
- Classical "rule of thirds" for facial vertical division
- Face theoretically divided into three equal vertical segments

**Eye Width / Face Width = 0.25**
- "Five-eye rule": face width equals five eye widths
- Each eye occupies 1/5 = 0.20 of face width, adjusted to 0.25 for measurement practicality

**Nose Width / Face Width = 0.20**
- Nose width as one-fifth of total face width
- Based on classical proportion studies

### 3. Angular Measurements

**Jaw Angle: 125 degrees**
- Average jawline angle from anthropometric studies
- Measured between left jaw-chin-right jaw points

**Eyebrow Angles: 45 degrees**
- Theoretical ideal eyebrow arch angle
- Based on classical facial proportion guidelines

### 4. Symmetry Analysis

**Midline Deviation**
- Calculates point-to-line distances from facial midline
- Uses standard deviation of distances, normalized by face width
- Converts to percentage deviation from perfect symmetry

## Scoring Methodology

### Root Mean Square (RMS) Deviation
The geometric harmony index uses RMS calculation:

```
For each ratio:
deviation = |measured_value - ideal_value| / ideal_value

RMS = √(mean(deviations²))
Score = max(0, 100 × (1 - RMS))
```

**Why RMS?**
- Penalizes large deviations more heavily than small ones
- Provides single composite score from multiple measurements
- Mathematically robust for comparing proportional relationships
- Range: 0-100 where 100 = perfect adherence to all ideal ratios

## Usage

```python
from main import greek_phi

# Analyze image
annotated_img, ratios, score, face_ratio = greek_phi("path/to/image.jpg")

# View results
print(f"Geometric Harmony Index: {score}%")
print(f"Face Ratio: {face_ratio}")
```

## Output Visualizations

- **Red dots**: Key landmark points
- **Colored lines**: Measured distances and angles
- **Gray line**: Facial midline for symmetry analysis
- **Golden spiral**: Mathematical φ-based spiral overlay
- **Annotated measurements**: Visual representation of calculated ratios

## Limitations and Considerations

### Technical Limitations
- Single face detection only
- Requires clear, front-facing images
- Dependent on MediaPipe landmark accuracy
- 2D analysis of 3D structures

### Mathematical Limitations
- Ideal ratios are theoretical constructs, not biological constants
- Population variation in facial proportions is substantial
- Cultural and ethnic differences in typical proportions
- No scientific consensus on "ideal" facial proportions

### Research Context
Current research indicates:
- Limited evidence for universal golden ratio adherence in attractive faces
- Significant population variations in typical facial ratios
- Cultural differences in aesthetic preferences
- Mathematical ratios don't predict perceived attractiveness reliably

## Dependencies

```
cv2>=4.5.0
mediapipe>=0.8.9
numpy>=1.20.0
```

## Installation

```bash
pip install opencv-python mediapipe numpy
python facial_analysis.py
```

## Scientific References

The mathematical relationships analyzed are based on:
- Classical proportion theories from Renaissance art
- Anthropometric measurement standards
- Geometric analysis principles
- Limited research on golden ratio in facial analysis

**Disclaimer**: This tool is for educational and research purposes. Results should not be interpreted as assessments of attractiveness or used for any decision-making affecting individuals' well-being.