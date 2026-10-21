"""Optional removal of burned-in HUD overlays (logo, gauge, ...) from the frames DPVO sees.

Overlays are fixed to the image, so DPVO patches on them look like perfectly static scene points.
inpaint_stream_worker replaces tracking._stream_worker (as thermal.py does for CLAHE): the unchanged
starter ClipPreprocessor decodes and preprocesses each frame, then the overlay rectangles are
filled with cv2.inpaint (TELEA), so the fill takes on the surrounding scene instead of becoming a
new static black box. Rectangles are given in full-resolution encoded-video pixels
(x0, y0, x1, y1 inclusive) through the FERMARBLE_OVERLAY_RECTS environment variable (JSON), which
the spawned decoder process inherits. The starter files stay unchanged.
"""
from __future__ import annotations
import json
import os

import cv2
import numpy as np

ENV = 'FERMARBLE_OVERLAY_RECTS'
MARGIN_PX = 16


def processed_mask(rects, pre, shape):
    """Overlay mask in the preprocessed frame (active crop, half resolution, trimmed)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    s = 0.5 if pre.half_res else 1.0
    for x0, y0, x1, y1 in rects:
        x0, x1 = (x0 - MARGIN_PX - pre.region.x0) * s, (x1 + MARGIN_PX - pre.region.x0) * s
        y0, y1 = (y0 - MARGIN_PX - pre.region.y0) * s, (y1 + MARGIN_PX - pre.region.y0) * s
        xa, xb = int(max(0, np.floor(x0))), int(min(w - 1, np.ceil(x1)))
        ya, yb = int(max(0, np.floor(y0))), int(min(h - 1, np.ceil(y1)))
        if xa <= xb and ya <= yb:
            mask[ya:yb + 1, xa:xb + 1] = 255
    return mask


def inpaint_stream_worker(queue, video_path, preset, target_fps, calibration=None, frame_timestamps=None):
    """Drop-in for starter tracking._stream_worker (same queue protocol) that inpaints overlays."""
    from preprocessing import ClipPreprocessor
    try:
        rects = json.loads(os.environ.get(ENV, '[]'))
        pre = ClipPreprocessor(video_path, preset, target_fps=target_fps,
                               calibration=calibration, frame_timestamps=frame_timestamps)
        mask = None
        for index, image, intrinsics, pts, decoded in pre.frames_with_time():
            if rects:
                if mask is None:
                    mask = processed_mask(rects, pre, image.shape)
                image = np.ascontiguousarray(cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA))
            queue.put((index, image, intrinsics, pts, decoded))
        queue.put((-1, None, None, -1.0, -1))
    except Exception as exc:
        queue.put((-2, str(exc), None, -1.0, -1))


def install(rects):
    """Make starter tracking.track() decode through inpaint_stream_worker with these rectangles."""
    import tracking
    os.environ[ENV] = json.dumps([list(map(int, r)) for r in rects])
    tracking._stream_worker = inpaint_stream_worker
