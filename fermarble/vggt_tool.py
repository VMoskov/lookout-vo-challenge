#!/usr/bin/env python3
"""VGGT-1B (external pretraining, facebook/VGGT-1B, non-commercial checkpoint) on keyframes of a clip:
per-window focal length / field of view, and chained camera rotations as a heading source.

Run with the VGGT environment (~/venvs/vggt/bin/python) from the repository root. Never reads
references.

Per clip:
- Keyframes every ~0.5 s (starter ClipPreprocessor at --keyframe-fps, the same active crop and
  half resolution as DPVO; CLAHE on thermal clips as in run_dpvo.py --clahe), resized to 518 px wide
  and centre-cropped to a multiple of 14 rows (cropping rows keeps fx unchanged).
- Windows of --window keyframes overlapping by --overlap; VGGT aggregator + camera head only.
- Field of view per window: median over its frames of 2*atan(518 / (2 fx)). The implied focal
  scale vs the released calibration is tan(nominal/2) / tan(vggt/2).
- Chaining: each window's world is its first camera. Window k+1 is rotated (chordal mean over the
  overlap) and similarity-aligned (overlap camera centres) onto the chain so far.
Writes <output>/poses/<clip>.npz in run_dpvo.py's pose format (camera-to-world tx ty tz qx qy qz qw at
keyframe times, with frame indices), plus per-window focal data and <output>/vggt_summary.json.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_DATASET, asset_path, load_manifest, load_timestamps, rotmat_to_quat,  # noqa: E402
                    select_clips, write_json)
import thermal  # noqa: E402

VGGT_WIDTH = 518


def shifted_frames(pre, frame_times, phase):
    """Like ClipPreprocessor.frames_with_time (every stride-th decoded frame), with the kept frames
    shifted later by round(phase * stride) decoded frames."""
    off = int(round(phase * pre.stride))
    cap = cv2.VideoCapture(pre.video_path)
    try:
        idx = -1
        while True:
            ok, image = cap.read()
            if not ok:
                break
            idx += 1
            if idx >= off and (idx - off) % pre.stride == pre.stride - 1 and idx < len(frame_times):
                yield None, pre.transform(image), None, float(frame_times[idx]), idx
    finally:
        cap.release()


def keyframes(root, clip, fps, phase=0.0, crop=0.0, mask_path=None):
    from preprocessing import ClipPreprocessor
    t = load_timestamps(root, clip)
    pre = ClipPreprocessor(str(asset_path(root, clip['video'])), clip['camera_mode'], fps,
                           calibration=clip['calibration'], frame_timestamps=t)
    clahe = cv2.createCLAHE(thermal.CLAHE_CLIP_LIMIT, thermal.CLAHE_TILE_GRID) if clip['camera_mode'] in thermal.THERMAL_MODES else None
    imgs, times, dec = [], [], []
    inpaint = None
    if mask_path:
        # mask in active-rect coordinates (any size): to the preprocessed frame = half the active
        # rect, trimmed at the bottom/right like the starter does
        x0, y0, x1, y1 = clip['calibration']['active_rect']
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        inpaint = (cv2.resize(m, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
    for _, img, _, pts, d in (shifted_frames(pre, t, phase) if phase else pre.frames_with_time()):
        if clahe is not None:
            img = thermal.apply_clahe(img, clahe)
        if inpaint is not None:   # same inpainting as the HUD test (overlay.py)
            img = cv2.inpaint(img, inpaint[:img.shape[0], :img.shape[1]], 5, cv2.INPAINT_TELEA)
        if crop:
            img = thermal.crop_bottom(img, crop, multiple=1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        nh = round(h * VGGT_WIDTH / w)
        r = cv2.resize(rgb, (VGGT_WIDTH, nh), interpolation=cv2.INTER_AREA)
        ch = nh // 14 * 14
        o = (nh - ch) // 2
        imgs.append(np.ascontiguousarray(r[o:o + ch]))
        times.append(pts)
        dec.append(d)
    nominal = math.degrees(2 * math.atan(pre.intrinsics.width / 2 / pre.intrinsics.fx))
    sx, sy = VGGT_WIDTH / w, nh / h                         # released intrinsics in the resized, cropped keyframe
    K = np.array([[pre.intrinsics.fx * sx, 0, pre.intrinsics.cx * sx],
                  [0, pre.intrinsics.fy * sy, pre.intrinsics.cy * sy - o], [0, 0, 1]])
    return np.stack(imgs), np.array(times), np.array(dec), nominal, len(t), K


def run_window(model, frames, K=None):
    """VGGT on one window; K (intrinsics injection) is not supported and ignored."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div(255)[None]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok, _ = model.aggregator(x)
        pe = model.camera_head(tok)[-1]
    ext, K = pose_encoding_to_extri_intri(pe.float(), x.shape[-2:])
    ext = ext[0].cpu().double().numpy()                    # world (= first camera) to camera, 3x4
    R_wc = np.transpose(ext[:, :, :3], (0, 2, 1))         # camera-to-world rotations
    centres = -np.einsum('nij,nj->ni', R_wc, ext[:, :, 3])
    fx = K[0, :, 0, 0].cpu().double().numpy()
    return R_wc, centres, fx


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def process_clip(model, root, clip, fps, window, overlap, run=run_window, inject_scale=None, phase=0.0, crop=0.0,
                 mask_path=None):
    """Chained poses and per-window FOV of one clip. `run(model, frames, K)` returns one window's
    camera-to-world rotations, centres and fx; with `inject_scale` it gets the released keyframe
    intrinsics with fx, fy scaled by it (else K=None). `phase` (0..1) shifts the keyframes by that
    fraction of the keyframe interval; `crop` drops that fraction of each frame at the bottom;
    `mask_path` inpaints that mask (static own-boat structure) first."""
    t0 = time.time()
    frames, times, dec, nominal, n_frames, K = keyframes(root, clip, fps, phase, crop, mask_path)
    K = np.diag([inject_scale, inject_scale, 1.0]) @ K if inject_scale is not None else None
    n = len(frames)
    R = np.full((n, 3, 3), np.nan)
    C = np.full((n, 3), np.nan)
    disp = np.full((n, 3), np.nan)   # displacement from the previous keyframe in its window's own scale (unchained)
    hfov_w, centre_t = [], []
    starts = list(range(0, max(1, n - overlap), window - overlap))
    for s in starts:
        e = min(n, s + window)
        if e - s < max(3, overlap + 1):
            break
        Rw, Cw, fx = run(model, frames[s:e], K)
        hfov_w.append(float(np.median(np.degrees(2 * np.arctan(VGGT_WIDTH / 2 / fx)))))
        centre_t.append(float(times[(s + e - 1) // 2]))
        if s == 0:
            R[s:e], C[s:e] = Rw, Cw
            disp[s + 1:e] = np.diff(Cw, axis=0)
            continue
        ov = np.arange(s, min(s + overlap, e))
        k = ov - s
        G = project_so3(sum(R[i] @ Rw[j].T for i, j in zip(ov, k)))   # window world -> chain world
        a, b = Cw[k] @ G.T, C[ov]                                      # similarity on overlap centres
        da, db = a - a.mean(0), b - b.mean(0)
        sc = float(np.sqrt((db ** 2).sum() / (da ** 2).sum())) if (da ** 2).sum() > 1e-12 else 1.0
        tr = b.mean(0) - sc * a.mean(0)
        new = np.arange(s + len(ov), e)
        R[new] = np.einsum('ij,njk->nik', G, Rw[new - s])
        C[new] = sc * (Cw[new - s] @ G.T) + tr
        disp[new] = (Cw[new - s] - Cw[new - s - 1]) @ G.T
    ok = np.isfinite(R).all((1, 2))
    poses = np.full((n, 7), np.nan)
    poses[ok, :3] = C[ok]
    poses[ok, 3:] = rotmat_to_quat(R[ok])
    return dict(poses=poses, times=times, frame_indices=dec, tracker_indices=np.arange(n),
                n_frames=np.int64(n_frames), window_hfov_deg=np.array(hfov_w), window_time=np.array(centre_t),
                nominal_hfov_deg=np.float64(nominal), keyframe_fps=np.float64(fps),
                window=np.int64(window), overlap=np.int64(overlap), window_disp=disp), time.time() - t0


def build_parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--keyframe-fps', type=float, default=2.0)
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--phase', type=float, default=0.0, help='shift keyframes by this fraction of their interval')
    ap.add_argument('--crops', type=Path, help='JSON {clip: fraction}: drop that fraction of each frame at the bottom')
    ap.add_argument('--edits', type=Path, help='JSON {clip: {"crop": fraction, "mask": png}} (structure_masks.py): '
                                               'inpaint the mask, then crop; overrides --crops for its clips')
    return ap


def run_clips(model, args, run=run_window, inject_scale=None, name='vggt'):
    """process_clip on every selected clip (done clips are skipped); writes <output>/poses/<clip>.npz
    and <output>/<name>_summary.json."""
    root = args.dataset.resolve().parent
    clips = select_clips(load_manifest(args.dataset), args.split, args.clips)
    out = args.output / 'poses'
    out.mkdir(parents=True, exist_ok=True)
    summ_path = args.output / f'{name}_summary.json'
    summary = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
    crops = json.loads(args.crops.read_text()) if args.crops else {}
    edits = json.loads(args.edits.read_text()) if args.edits else {}
    for c in clips:
        if (out / f"{c['key']}.npz").is_file():
            print(c['key'], 'done, skipping', flush=True)
            continue
        try:
            ed = edits.get(c['key'], {})
            crop = float(ed.get('crop', crops.get(c['key'], 0.0)))
            res, secs = process_clip(model, root, c, args.keyframe_fps, args.window, args.overlap, run, inject_scale,
                                     args.phase, crop, ed.get('mask'))
            if ed.get('mask'):
                res['inpaint_mask'] = np.array(ed['mask'])
            if crop:
                res['crop_bottom'] = np.float64(crop)
            if args.phase:
                res['keyframe_phase'] = np.float64(args.phase)
            if inject_scale is not None:
                res['inject_focal_scale'] = np.float64(inject_scale)
            np.savez(out / f"{c['key']}.npz", **res)
            h = res['window_hfov_deg']
            nom = float(res['nominal_hfov_deg'])
            med = float(np.median(h))
            summary[c['key']] = dict(camera_mode=c['camera_mode'], split=c['split'], n_keyframes=len(res['times']),
                                     windows=len(h), nominal_hfov_deg=nom, **{f'{name}_hfov_median': med,
                                     f'{name}_hfov_iqr': [float(np.percentile(h, 25)), float(np.percentile(h, 75))]},
                                     focal_scale=math.tan(math.radians(nom) / 2) / math.tan(math.radians(med) / 2),
                                     finite_rot=float(np.isfinite(res['poses']).all(1).mean()), seconds=round(secs, 1))
        except Exception as exc:
            summary[c['key']] = dict(camera_mode=c['camera_mode'], split=c['split'], error=str(exc)[:300])
        write_json(summ_path, summary)
        print(c['key'], summary[c['key']], flush=True)
    return 0


def main(argv=None):
    args = build_parser(__doc__).parse_args(argv)
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval()
    return run_clips(model, args)


if __name__ == '__main__':
    raise SystemExit(main())
