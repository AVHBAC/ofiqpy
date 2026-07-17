"""Session container — the shared intermediate products OFIQ computes once per image.

Mirrors OFIQ's Session/FaceImage: the pipeline fills these fields in order
(detect -> landmarks -> align -> pose -> parse -> occlusion), and measures read
from them. Kept deliberately close to OFIQ's OFIQImpl preprocessing products so
each measure consumes the same inputs the reference does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Session:
    image: np.ndarray  # original BGR uint8 (H, W, 3)

    # detection
    bbox: np.ndarray | None = None            # (4,) [x1, y1, x2, y2] in original px
    n_faces: int = 0
    face_areas: list[float] = field(default_factory=list)  # descending

    # landmarks (ADNet 98-pt), original-image pixel coords
    landmarks: np.ndarray | None = None       # (98, 2)

    # alignment (616x616 canonical) + transformed landmarks
    aligned_face: np.ndarray | None = None    # (616, 616, 3) BGR uint8
    aligned_landmarks: np.ndarray | None = None  # (98, 2) in aligned space
    affine: np.ndarray | None = None          # (2, 3) transform used

    # pose (degrees)
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None

    # segmentation
    parsing: np.ndarray | None = None         # aligned parsing label map
    occlusion_mask: np.ndarray | None = None  # aligned binary occlusion mask

    # derived masks
    landmarked_region: np.ndarray | None = None  # aligned face-region mask

    def has(self, *attrs: str) -> bool:
        return all(getattr(self, a) is not None for a in attrs)
