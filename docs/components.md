# Components

ofiqpy computes all 27 ISO/IEC 29794-5 quality components plus the unified quality score.
Each is ported line-faithfully from the corresponding OFIQ measure; the CSV column names
match `OFIQSampleApp`.

## Capture / pixel measures

| Component | ISO § | Native value | Mapping |
|---|---|---|---|
| BackgroundUniformity | 7.3.2 | mean Scharr gradient over the eroded background mask | sigmoid |
| IlluminationUniformity | 7.3.3 | histogram intersection of the two cheek ROIs | `100·D^0.3` |
| LuminanceMean | 7.3.4.2 | histogram mean of face-region luminance | double-sigmoid |
| LuminanceVariance | 7.3.4.3 | histogram variance | `100·sin(...)` |
| UnderExposurePrevention | 7.3.5 | fraction of Y≤25 in region ∩ non-occluded | sigmoid |
| OverExposurePrevention | 7.3.6 | fraction of Y≥247 in region ∩ non-occluded | `1/(v+0.01)` |
| DynamicRange | 7.3.7 | Shannon entropy of region luminance | `12.5·entropy` |
| Sharpness | 7.3.8 | RTrees on 26 gradient features (original image) | sigmoid |
| CompressionArtifacts | 7.3.9 | ssim-248 CNN on the aligned 248² crop | sigmoid |
| NaturalColour | 7.3.10 | CIELAB a*/b* distance from the skin plateau | sigmoid |

## Subject / geometry measures

| Component | ISO § | Native value | Mapping |
|---|---|---|---|
| SingleFacePresent | 7.4.2 | 2nd-largest / largest face area | `100·(1−f)` |
| EyesOpen | 7.4.3 | min eyelid opening / tmetric (aligned) | sigmoid |
| MouthClosed | 7.4.4 | max inner-lip opening / tmetric | sigmoid |
| EyesVisible | 7.4.5 | occluded fraction of the eye zones | `100·(1−α)` |
| MouthOcclusionPrevention | 7.4.6 | occluded fraction of the mouth polygon | `100·(1−α)` |
| FaceOcclusionPrevention | 7.4.7 | occluded fraction of the landmarked region | `100·(1−α)` |
| InterEyeDistance | 7.4.8 | eye-center distance / cos(yaw) | sigmoid |
| HeadSize | 7.4.9 | tmetric / image height | sigmoid |
| LeftwardCropOfTheFaceImage | 7.4.10.1 | right-eye x / IED | sigmoid |
| RightwardCropOfTheFaceImage | 7.4.10.2 | (width − left-eye x) / IED | sigmoid |
| MarginAboveOfTheFaceImage | 7.4.10.3 | eye-midpoint y / t | sigmoid |
| MarginBelowOfTheFaceImage | 7.4.10.4 | (height − eye-midpoint y) / t | sigmoid |
| HeadPoseYaw | 7.4.11 | geometric pitch (OFIQ slot swap) | `100·cos²` |
| HeadPosePitch | 7.4.11 | geometric yaw (OFIQ slot swap) | `100·cos²` |
| HeadPoseRoll | 7.4.11 | geometric roll | `100·cos²` |
| ExpressionNeutrality | 7.4.12 | AdaBoost score over dual-EfficientNet embeddings | sigmoid |
| NoHeadCoverings | 7.4.13 | fraction of cloth/hat pixels (top crop) | piecewise |

## Unified

| Component | Native value | Mapping |
|---|---|---|
| UnifiedQualityScore | MagFace embedding magnitude | `sigmoid(x0=23, w=2.6)` |

!!! note "GazeDirection (C28)"
    OFIQ does not compute gaze; ofiqpy follows OFIQ and omits it from the standard output.
