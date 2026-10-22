#!/usr/bin/env python3
"""Pi3X (external pretraining, yyfz233/Pi3X) on keyframes of a clip: per-window focal length /
field of view, and chained camera rotations as a heading source.

Same keyframes, windows, chaining and output format as vggt_tool.py (only the per-window model
differs), so make_predictions.py --vggt and the oracle take a Pi3X run directory unchanged.
Run with the Pi3 environment (~/venvs/pi3/bin/python) from the repository root. The Pi3 checkout
(github.com/yyfz/Pi3) is found through PI3_DIR (default ../Pi3). Never reads references.

- Poses: Pi3X camera_poses (camera-to-world, OpenCV axes).
- Focal: Pi3X predicts per-pixel local point maps; their ray directions give fx per frame through
  the checkout's recover_intrinsic_from_rays_d (principal point at the image centre).
- --inject-focal S: condition Pi3X on the released intrinsics of the keyframes with fx, fy
  multiplied by S (Pi3X's multimodal branch); without it the multimodal branch is disabled.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vggt_tool import build_parser, run_clips  # noqa: E402

PI3_DIR = Path(os.environ.get('PI3_DIR', Path(__file__).resolve().parents[2] / 'Pi3'))
sys.path.insert(0, str(PI3_DIR))


def run_window(model, frames, K=None):
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div(255)[None]
    cond = {}
    if K is not None:
        cond['intrinsics'] = torch.from_numpy(np.repeat(K[None], len(frames), 0)).float().cuda()[None]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        res = model(imgs=x, **cond)
    T = res['camera_poses'][0].float().cpu().double().numpy()            # camera-to-world 4x4
    rays = torch.nn.functional.normalize(res['local_points'][0].float(), dim=-1)
    fx = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)[:, 0, 0].cpu().double().numpy()
    return T[:, :3, :3], T[:, :3, 3], fx


def main(argv=None):
    ap = build_parser(__doc__)
    ap.add_argument('--inject-focal', type=float, help='condition on released intrinsics with fx, fy x this')
    args = ap.parse_args(argv)
    from pi3.models.pi3x import Pi3X
    model = Pi3X.from_pretrained('yyfz233/Pi3X').eval()
    if args.inject_focal is None:
        model.disable_multimodal()
    model = model.to('cuda')
    return run_clips(model, args, run_window, args.inject_focal, name='pi3x')


if __name__ == '__main__':
    raise SystemExit(main())
