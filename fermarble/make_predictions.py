#!/usr/bin/env python3
"""CPU step: DPVO camera heading + constant speed -> starter-format predictions for every frame.

For each selected clip, reads <poses>/poses/<clip>.npz from run_dpvo.py. The heading is the
camera forward axis (+z) projected onto the plane perpendicular to up, where up is minus the
mean camera +y axis. It is unwrapped, Gaussian-smoothed in time and interpolated to EVERY
released frame time. It is then integrated at constant speed into a planar path with z = 0.
Clips without usable poses get a straight line, so every selected clip gets a valid
prediction. Never reads references.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import (DEFAULT_DATASET, SMOOTH_SIGMA_S, heading_constant_speed, load_manifest,
                    load_pose_npz, load_timestamps, prediction_dict, select_clips,
                    validate_prediction, write_json)


def predict_clip(root, clip, pose_path, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """Returns (prediction dict, info dict) for one clip."""
    frame_times = load_timestamps(root, clip)
    poses = pose_times = None
    info = {}
    if pose_path is not None and pose_path.is_file():
        try:
            data = load_pose_npz(pose_path)
            poses, pose_times = data['poses'], data['times']
            if not np.allclose(pose_times, frame_times[data['frame_indices']], rtol=0, atol=1e-6):
                raise ValueError('pose times do not match released frame timestamps')
        except Exception as exc:  # corrupt/mismatched npz -> fallback, but say why
            poses = pose_times = None
            info['pose_load_error'] = str(exc)[:200]
    positions, model_info = heading_constant_speed(poses, pose_times, frame_times, sigma_s, invert)
    info.update(model_info)
    if pose_path is None or not pose_path.is_file():
        info['reason'] = 'no pose file'
    pred = prediction_dict(clip['key'], frame_times, positions)
    validate_prediction(pred, clip['key'], frame_times)
    return pred, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val', help='train, val, test, comma-separated, or all')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--poses', type=Path, help='run_dpvo.py output directory (omit: all fallback)')
    ap.add_argument('--output', required=True, type=Path, help='new predictions run directory')
    ap.add_argument('--sigma', type=float, default=SMOOTH_SIGMA_S, help='heading smoothing sigma, s')
    ap.add_argument('--invert-poses', action='store_true', help='convention check: treat poses as world-to-camera')
    ap.add_argument('--overwrite', action='store_true', help='allow a non-empty --output')
    args = ap.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        ap.error('--output must be new or empty (or pass --overwrite)')
    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    root = args.dataset.resolve().parent
    records = {}
    for clip in selected:
        key = clip['key']
        pose_path = args.poses / 'poses' / f'{key}.npz' if args.poses else None
        pred, info = predict_clip(root, clip, pose_path, args.sigma, args.invert_poses)
        write_json(args.output / 'predictions' / f'{key}.json', pred)
        records[key] = dict(split=clip['split'], camera_mode=clip['camera_mode'],
                            n_frames=len(pred['times']), **info)
        print(f"{key}: {info['method']}" + (f" ({info['reason']})" if 'reason' in info else ''), flush=True)
    write_json(args.output / 'prediction_manifest.json',
               dict(poses=str(args.poses) if args.poses else None, sigma_s=args.sigma,
                    invert_poses=args.invert_poses, clips=records))
    counts = Counter(r['method'] for r in records.values())
    print(f'PREDICTIONS_COMPLETE: {len(records)} clips, {dict(counts)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
