#!/usr/bin/env python3
"""Segmentation preview, stage A (DPVO env, CPU, no references): sample frames + flow-band geometry.

For each clip, 3 frames (15 / 50 / 85 % of the clip) exactly as DPVO and flow_speed.py see them
(starter ClipPreprocessor at 8 fps: active crop, half resolution, trim), in colour, plus the
geometry flow_speed.py uses on that frame: the fitted horizon (DPVO up + per-window pitch/roll
correction) and the water sample points between MIN/MAX_DEPRESSION_DEG below it, split into the
near and far bands at the median depression. Clips without a flow file get frames only.
Writes <output>/<clip>_<i>.npz for seg_preview.py (stage B, PanSR env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import asset_path, estimate_up, load_manifest, load_timestamps, quat_to_rotmat, select_clips  # noqa: E402
from flow_speed import GRID_STEP, best_pose_run, correction, horizon_rows, water_region  # noqa: E402

SETS = {  # manifest, pose dirs (as in the pipeline), flow dir
    'dev': ('release/release_manifest.json', [f'runs/dpvo-fps8-seed{s}' for s in range(3)], 'runs/flow-v2'),
    'test': ('lookout-vo-test-v0.1/release_manifest.json', [f'runs/merged-test-seed{s}-clahe' for s in range(3)],
             'runs/flow-test'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dev', default='clip_021,clip_050,clip_075,clip_084,clip_028,clip_089,clip_055,clip_046')
    ap.add_argument('--test', default='clip_093,clip_104,clip_108,clip_109,clip_100,clip_106')
    ap.add_argument('--output', type=Path, default=Path('runs/seg-preview/frames'))
    args = ap.parse_args()
    from preprocessing import ClipPreprocessor  # starter
    args.output.mkdir(parents=True, exist_ok=True)
    for split, keys in (('dev', args.dev), ('test', args.test)):
        mpath, pose_dirs, flow_dir = SETS[split]
        mpath = Path(mpath)
        for c in select_clips(load_manifest(mpath), 'all', keys):
            k = c['key']
            ft = load_timestamps(mpath.parent, c)
            pre = ClipPreprocessor(str(asset_path(mpath.parent, c['video'])), c['camera_mode'], 8.0,
                                   calibration=c['calibration'], frame_timestamps=ft)
            frames = list(pre.frames_with_time())
            times = np.array([f[3] for f in frames])
            want = [int(np.argmin(np.abs(times - q))) for q in np.quantile(times, [0.15, 0.5, 0.85])]
            fx, fy, cx, cy = (float(v) for v in frames[0][2])
            flow = Path(flow_dir) / 'speed' / f'{k}.npz'
            geo = None
            if flow.is_file():
                d = dict(np.load(flow))
                z, _ = best_pose_run([Path(p) / 'poses' / f'{k}.npz' for p in pose_dirs])
                R = quat_to_rotmat(np.where(np.isfinite(z['poses'][:, 3:7]), z['poses'][:, 3:7], [0, 0, 0, 1]))
                U = estimate_up(R[np.isfinite(z['poses']).all(1)])
                geo = (d, R, U)
            for i, j in enumerate(want):
                _, img, _, pts, dec = frames[j]
                H, W = img.shape[:2]
                out = dict(image=img, key=k, split=split, camera_mode=c['camera_mode'], time=pts, decoded=dec,
                           K=np.array([fx, fy, cx, cy]), has_band=geo is not None)
                if geo is not None:
                    d, R, U = geo
                    u = correction(d['horizon_pitch_deg'][j], d['horizon_roll_deg'][j]) @ (R[j].T @ U)
                    u /= np.linalg.norm(u)
                    gy, gx = np.mgrid[GRID_STEP // 2:H:GRID_STEP, GRID_STEP // 2:W:GRID_STEP]
                    p = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
                    rays = np.column_stack([(p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, np.ones(len(p))])
                    water, s = water_region(rays, u, np.zeros(len(p), bool))
                    split_s = np.median(s[water]) if water.any() else 0.0
                    xs = np.arange(0, W, 2.0)
                    out.update(near=p[water & (s >= split_s)], far=p[water & (s < split_s)],
                               horizon=np.column_stack([xs, horizon_rows(u, (fx, fy, cx, cy), xs)]))
                np.savez_compressed(args.output / f'{k}_{i}.npz', **out)
            print(k, c['camera_mode'], 'band' if geo is not None else 'no flow band', flush=True)


if __name__ == '__main__':
    main()
