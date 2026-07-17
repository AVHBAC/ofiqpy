# Architecture

ofiqpy mirrors OFIQ's `OFIQImpl` preprocessing order: each stage's product is stored on a
`Session` and reused by the measures.

```
image (BGR)
  │
  ├─ SSD face detector (cv2.dnn, Caffe) ─────────► bounding box, face areas
  │
  ├─ 3DDFA-V2 head pose (ONNX) ──────────────────► yaw, pitch, roll
  │
  ├─ ADNet-98 landmarks (ONNX) ──────────────────► 98 landmarks (original px)
  │        │
  │        └─ 5-point similarity alignment (LMEDS) ► aligned face 616×616,
  │                                                  aligned landmarks, affine M
  │
  ├─ BiSeNet face parsing (ONNX, on aligned) ────► 400×400 class map
  ├─ face-occlusion segmentation (ONNX, aligned) ► 616×616 occlusion mask
  └─ landmarked-region mask (GetFaceMask α=0) ───► 616×616 face-region mask
                                                   │
                                                   ▼
                                     28 measures ► {name: (raw, scalar)}
```

## Backbone models (all OFIQ's own weights)

| Stage | Model | Engine |
|---|---|---|
| Detection | `ssd_facedetect.caffemodel` (ResNet-SSD) | OpenCV DNN |
| Landmarks | `ADNet.onnx` (98-point) | onnxruntime |
| Pose | `mb1_120x120.onnx` (3DDFA-V2) | onnxruntime |
| Face parsing | `bisenet_400.onnx` (BiSeNet) | onnxruntime |
| Occlusion | `face_occlusion_segmentation_ort.onnx` | onnxruntime |
| Sharpness | `face_sharpness_rtree.xml.gz` (random forest) | `cv2.ml.RTrees` |
| Compression | `ssim_248_model.onnx` (CNN) | onnxruntime |
| Expression | `enet_b0` + `enet_b2` (HSEmotion) + AdaBoost | onnxruntime + `cv2.ml.Boost` |
| Unified | `magface_iresnet50_norm.onnx` (MagFace) | onnxruntime |

## Faithfulness details

Reproducing OFIQ to ±1 depends on several exact behaviors, each verified against the C++:

- **Alignment**: 5 source points (eye centers of ADNet corners 60/64 & 68/72, nose 54,
  mouth 76/82) → fixed reference points, `estimateAffinePartial2D(..., LMEDS)`, `warpAffine`
  to 616×616.
- **ADNet back-projection**: landmarks scaled back by `squareBox.height / 256` (the
  `floor`/`ceil` squaring can leave the box 1px non-square — the *height* is authoritative).
- **HeadPose slot swap**: OFIQ's `HeadPoseYaw` output reports the geometric *pitch* and
  vice-versa (`HeadPose.cpp`); ofiqpy reproduces the swap.
- **Sigmoid**: `quality = h·(a + s·sigmoid(x; x0, w))`, C-style round (half away from zero),
  clamp `[0, 100]`. Several measures use non-sigmoid maps (LuminanceVariance = sin,
  OverExposure = `1/(v+0.01)`, DynamicRange = `12.5·entropy`).
- **Colour order per model** differs (pose/UQS = BGR; parsing/occlusion/compression = RGB).

## Package layout

```
ofiqpy/
  config.py          JAXN config loader + OFIQ model resolver (env-driven)
  session.py         shared preprocessing products
  pipeline.py        the OFIQImpl-order orchestrator
  sigmoid.py         OFIQ ScalarConversion
  detectors/ssd.py   SSD detector (OpenCV DNN)
  landmarks/adnet.py ADNet-98 + square-crop helpers
  align.py           alignment, landmarked region, tmetric, luminance
  pose/tddfa.py      3DDFA-V2 pose
  segmentation/      BiSeNet parsing, occlusion segmentation
  measures/          core (dispatch + model cache), geometry, pixel, models, helpers
  output.py          OFIQ-format CSV
  cli.py, batch.py   single + parallel runners
```
