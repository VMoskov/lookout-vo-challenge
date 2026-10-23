#!/usr/bin/env python3
"""PanSR (LaRS-pretrained Swin-L, external pretraining, github.com/lojzezust/PanSR, Apache-2.0)
segmentation of the frames flow_speed.py uses (GPU, PanSR env; never reads references).

Per clip, the starter ClipPreprocessor frames at 8 fps (active crop, half resolution, trim) exactly
as DPVO and flow see them; every --stride-th frame is segmented at its native size (short side
MIN_SIZE, no upscaling: water masks agree 99.6 % with PanSR's default 1024 px on a preview sample,
at ~3.5x the speed). Decoding runs in a thread so the GPU stays busy. Writes
<output>/<clip>.npz (resumable, one file per clip):
  frame_indices, times     all 8 fps frames (decoded index, time), as in flow_speed.py
  seg_frames               indices into those frames that were segmented
  masks                    LaRS class id per pixel at 1/QUARTER resolution (uint8; 0 void,
                           1-8 dynamic obstacles, 9 static obstacle, 10 water, 11 sky)
  water_top                per column (full processed resolution), the top row of the largest
                           connected water region (-1 where the column has none)
Run with ~/micromamba/envs/pansr/bin/python from lookout-vo-challenge.
"""
from __future__ import annotations
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'jobs'))
from common import DEFAULT_DATASET, asset_path, load_manifest, load_timestamps, select_clips  # noqa: E402

WATER = 10
QUARTER = 4
MIN_SIZE = 528


def load_predictor(repo):
    from detectron2.engine.defaults import DefaultPredictor
    from seg_preview import load_predictor as preview_predictor
    pr = preview_predictor(repo)
    cfg = pr.cfg.clone()
    cfg.defrost()
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = MIN_SIZE, 2 * MIN_SIZE
    cfg.freeze()
    out = DefaultPredictor(cfg)
    out.model = pr.model
    return out


def segment(predictor, img):
    with torch.no_grad():
        pan, info = predictor(img)['panoptic_seg']
    lut = torch.zeros(int(pan.max()) + 1, dtype=torch.uint8, device=pan.device)
    for s in info:
        lut[s['id']] = s['category_id']
    return lut[pan.long()].cpu().numpy()


def water_top(cls):
    """Per column, the top row of the largest connected water region (-1 where none)."""
    water = (cls == WATER).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=4)
    top = np.full(cls.shape[1], -1, np.int16)
    if n < 2:
        return top
    big = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    has = big.any(0)
    top[has] = np.argmax(big[:, has], axis=0)
    return top


def process_clip(predictor, root, clip, out_path, stride):
    from preprocessing import ClipPreprocessor   # starter
    t0 = time.time()
    ft = load_timestamps(root, clip)
    pre = ClipPreprocessor(str(asset_path(root, clip['video'])), clip['camera_mode'], 8.0,
                           calibration=clip['calibration'], frame_timestamps=ft)
    q = queue.Queue(maxsize=64)

    def produce():
        try:
            for i, (_, img, _, pts, dec) in enumerate(pre.frames_with_time()):
                q.put((i, img if i % stride == 0 else None, pts, dec))
        finally:
            q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    dec_idx, times, seg_frames, masks, tops = [], [], [], [], []
    while (item := q.get()) is not None:
        i, img, pts, dec = item
        dec_idx.append(dec)
        times.append(pts)
        if img is None:
            continue
        cls = segment(predictor, img)
        seg_frames.append(i)
        masks.append(cv2.resize(cls, (cls.shape[1] // QUARTER, cls.shape[0] // QUARTER), interpolation=cv2.INTER_NEAREST))
        tops.append(water_top(cls))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, frame_indices=np.array(dec_idx), times=np.array(times), seg_frames=np.array(seg_frames),
                        masks=np.stack(masks), water_top=np.stack(tops), stride=np.int64(stride),
                        quarter=np.int64(QUARTER), min_size=np.int64(MIN_SIZE))
    tmp.rename(out_path)
    return len(dec_idx), len(seg_frames), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--modes', default='wide,flir-VIS', help='camera modes to segment (the flow modes)')
    ap.add_argument('--stride', type=int, default=2, help='segment every stride-th 8 fps frame')
    ap.add_argument('--model', default='lojzezust/pansr-lars-swinl')
    ap.add_argument('--output', type=Path, default=Path('runs/seg-pansr'))
    args = ap.parse_args(argv)
    root = args.dataset.resolve().parent
    clips = [c for c in select_clips(load_manifest(args.dataset), args.split, args.clips)
             if c['camera_mode'] in args.modes.split(',')]
    todo = [c for c in clips if not (args.output / f"{c['key']}.npz").is_file()]
    print(f'{len(todo)} of {len(clips)} clips to segment', flush=True)
    if not todo:
        return 0
    predictor = load_predictor(args.model)
    for n, c in enumerate(todo, 1):
        try:
            nf, ns, secs = process_clip(predictor, root, c, args.output / f"{c['key']}.npz", args.stride)
            print(f"[{n}/{len(todo)}] {c['key']} {c['camera_mode']}: {ns}/{nf} frames segmented in {secs:.0f} s", flush=True)
        except Exception as exc:
            print(f"[{n}/{len(todo)}] {c['key']} FAILED: {str(exc)[:300]}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
