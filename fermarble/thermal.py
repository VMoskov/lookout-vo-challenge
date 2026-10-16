"""Optional CLAHE contrast equalisation for thermal clips, applied to the frames DPVO sees.

The starter decodes and preprocesses frames in a spawned child process
(tracking._stream_worker), so a patch made in the parent never reaches the decoder. Instead,
install() swaps tracking._stream_worker for clahe_stream_worker below. The child imports this
module by name, runs the unchanged starter ClipPreprocessor, and equalises each preprocessed
frame (after crop, half-res and trim) when the clip's camera mode is in THERMAL_MODES.
The starter files stay unchanged.
"""
from __future__ import annotations
import sys

import cv2
import numpy as np

# flir-IR only: on train+val CLAHE helped flir-IR (3-seed median 10.8 -> 8.0 %) but hurt the
# 'infrared' clip (42.9 -> 52.4 %), which is a colour low-light camera, not thermal.
THERMAL_MODES = ('flir-IR',)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def apply_clahe(image, clahe):
    """CLAHE on a BGR uint8 frame. Grayscale-in-BGR frames (B = G = R, as flir-IR) are equalised
    directly; colour frames (the 'infrared' clip is a low-light colour camera) on LAB lightness."""
    if np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2]):
        return cv2.cvtColor(clahe.apply(image[..., 0]), cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_stream_worker(queue, video_path, preset, target_fps, calibration=None, frame_timestamps=None):
    """Drop-in for starter tracking._stream_worker (same queue protocol), plus CLAHE for thermal."""
    from preprocessing import ClipPreprocessor
    try:
        pre = ClipPreprocessor(video_path, preset, target_fps=target_fps,
                               calibration=calibration, frame_timestamps=frame_timestamps)
        clahe = None
        if preset in THERMAL_MODES:
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
            print(f'CLAHE active: {preset} clip, clip limit {CLAHE_CLIP_LIMIT}, tiles {CLAHE_TILE_GRID}',
                  file=sys.stderr, flush=True)
        for index, image, intrinsics, pts, decoded in pre.frames_with_time():
            if clahe is not None:
                image = np.ascontiguousarray(apply_clahe(image, clahe))
            queue.put((index, image, intrinsics, pts, decoded))
        queue.put((-1, None, None, -1.0, -1))
    except Exception as exc:
        queue.put((-2, str(exc), None, -1.0, -1))


def install():
    """Make starter tracking.track() decode through clahe_stream_worker."""
    import tracking
    tracking._stream_worker = clahe_stream_worker
