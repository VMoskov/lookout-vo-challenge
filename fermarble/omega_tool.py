#!/usr/bin/env python3
"""VGGT-Omega 1B (external pretraining: Meta FAIR VGGT-Omega, FAIR Noncommercial Research License; checkpoint =
OMEGA_CKPT: the official facebook/VGGT-Omega vggt_omega_1b_512.pt (gated access), or the backbone-finetuned copy in
huggingface.co/ChenmingWu/mapvggt, checkpoints/abl_base_best.safetensors, keys vggt.*, used with the author's
permission) on keyframes of a clip: per-window focal length / field of view and
chained camera rotations as a heading source.

Same keyframes, windows, chaining and output format as vggt_tool.py (only the per-window model differs), so
make_predictions.py --vggt and the oracle take an Omega run directory unchanged. Keyframes are resized to
OMEGA_WIDTH px wide (about Omega's 512^2-token budget at 16 px patches) and centre-cropped to a multiple of 16 rows.
Run with the VGGT environment (~/venvs/vggt/bin/python) from the repository root; the vggt-omega checkout
(github.com/facebookresearch/vggt-omega) is found through VGGT_OMEGA_DIR (default ../vggt-omega) and the checkpoint
through OMEGA_CKPT. Never reads references.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vggt_tool  # noqa: E402
from vggt_tool import build_parser, run_clips  # noqa: E402

OMEGA_DIR = Path(os.environ.get('VGGT_OMEGA_DIR', Path(__file__).resolve().parents[2] / 'vggt-omega'))
OMEGA_CKPT = Path(os.environ.get('OMEGA_CKPT', '/media/labust/shared3tb_ssd1/macvi5-weights/mapvggt/abl_base_best.safetensors'))
OMEGA_WIDTH = 704
sys.path.insert(0, str(OMEGA_DIR))


def load_model():
    """VGGTOmega (camera head only) with the vggt.* weights of the mapvggt checkpoint (.safetensors) or the official
    Meta state dict (.pt, loaded with weights_only); every aggregator and camera-head weight must be present."""
    from vggt_omega.models import VGGTOmega
    if OMEGA_CKPT.suffix == '.pt':
        sd = torch.load(OMEGA_CKPT, map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(str(OMEGA_CKPT))
        sd = {k[len('vggt.'):]: v for k, v in sd.items() if k.startswith('vggt.')}
    model = VGGTOmega(enable_depth=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or any(not k.startswith('dense_head.') for k in unexpected):
        raise RuntimeError(f'checkpoint mismatch: missing {missing[:5]}, unexpected {[k for k in unexpected if not k.startswith("dense_head.")][:5]}')
    return model.to('cuda').eval()


def run_window(model, frames, K=None):
    """Omega on one window (frames n x h x OMEGA_WIDTH x 3 uint8); K is not supported and ignored. Rows are
    centre-cropped to a multiple of 16 (fx unchanged)."""
    from vggt_omega.utils.pose_enc import encoding_to_camera
    h = frames.shape[1] // 16 * 16
    o = (frames.shape[1] - h) // 2
    x = torch.from_numpy(np.ascontiguousarray(frames[:, o:o + h])).cuda().permute(0, 3, 1, 2).float().div(255)[None]
    with torch.inference_mode():
        pe = model(x)['pose_enc']
    ext, Kp = encoding_to_camera(pe.float(), x.shape[-2:])
    ext = ext[0].cpu().double().numpy()                    # world (= first camera) to camera, 3x4
    R_wc = np.transpose(ext[:, :, :3], (0, 2, 1))         # camera-to-world rotations
    centres = -np.einsum('nij,nj->ni', R_wc, ext[:, :, 3])
    return R_wc, centres, Kp[0, :, 0, 0].cpu().double().numpy()


def main(argv=None):
    args = build_parser(__doc__).parse_args(argv)
    vggt_tool.VGGT_WIDTH = OMEGA_WIDTH       # keyframe width and the fx -> FOV conversion in vggt_tool
    return run_clips(load_model(), args, run_window, name='omega')


if __name__ == '__main__':
    raise SystemExit(main())
