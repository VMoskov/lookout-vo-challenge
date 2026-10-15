#!/usr/bin/env python3
"""CPU step: DPVO camera heading + constant speed -> starter-format predictions for every frame.

For each selected clip, reads <poses>/poses/<clip>.npz from run_dpvo.py (from several --poses
directories, e.g. one per seed, the headings are combined by their per-frame circular median). The heading is the
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
                    load_pose_npz, load_timestamps, multi_run_heading_constant_speed,
                    prediction_dict, select_clips, validate_prediction, write_json)


def predict_clip(root, clip, pose_paths, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """Returns (prediction dict, info dict) for one clip.

    `pose_paths` lists one npz per DPVO run (e.g. per seed); several runs are combined by the
    per-frame circular median of their headings. Missing or mismatched files are skipped.
    """
    frame_times = load_timestamps(root, clip)
    runs, errors = [], []
    for path in pose_paths or []:
        if not path.is_file():
            errors.append(f'{path}: no pose file')
            continue
        try:
            data = load_pose_npz(path)
            if not np.allclose(data['times'], frame_times[data['frame_indices']], rtol=0, atol=1e-6):
                raise ValueError('pose times do not match released frame timestamps')
            runs.append((data['poses'], data['times']))
        except Exception as exc:  # corrupt/mismatched npz -> skip this run, but say why
            errors.append(f'{path}: {str(exc)[:200]}')
    if invert:
        poses, times = runs[0] if runs else (None, None)
        positions, info = heading_constant_speed(poses, times, frame_times, sigma_s, invert=True)
    else:
        positions, info = multi_run_heading_constant_speed(runs, frame_times, sigma_s)
    if errors:
        info['pose_errors'] = errors
    if not runs:
        info['reason'] = 'no usable pose file'
    pred = prediction_dict(clip['key'], frame_times, positions)
    validate_prediction(pred, clip['key'], frame_times)
    return pred, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val', help='train, val, test, comma-separated, or all')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--poses', type=Path, nargs='+',
                    help='run_dpvo.py output directory; several (e.g. one per seed) are combined by '
                         'the circular median of their headings (omit: all fallback)')
    ap.add_argument('--output', required=True, type=Path, help='new predictions run directory')
    ap.add_argument('--sigma', type=float, default=SMOOTH_SIGMA_S, help='heading smoothing sigma, s')
    ap.add_argument('--invert-poses', action='store_true', help='convention check: treat poses as world-to-camera')
    ap.add_argument('--overwrite', action='store_true', help='allow a non-empty --output')
    args = ap.parse_args(argv)
    if args.invert_poses and len(args.poses or []) > 1:
        ap.error('--invert-poses takes a single --poses directory')
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        ap.error('--output must be new or empty (or pass --overwrite)')
    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    root = args.dataset.resolve().parent
    records = {}
    for clip in selected:
        key = clip['key']
        pose_paths = [d / 'poses' / f'{key}.npz' for d in args.poses or []]
        pred, info = predict_clip(root, clip, pose_paths, args.sigma, args.invert_poses)
        write_json(args.output / 'predictions' / f'{key}.json', pred)
        # JSON cannot hold NaN: a diagnostic that is undefined for a clip is recorded as null
        info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in info.items()}
        records[key] = dict(split=clip['split'], camera_mode=clip['camera_mode'],
                            n_frames=len(pred['times']), **info)
        print(f"{key}: {info['method']}" + (f" ({info['reason']})" if 'reason' in info else ''), flush=True)
    write_json(args.output / 'prediction_manifest.json',
               dict(poses=[str(d) for d in args.poses or []], sigma_s=args.sigma,
                    invert_poses=args.invert_poses, clips=records))
    counts = Counter(r['method'] for r in records.values())
    print(f'PREDICTIONS_COMPLETE: {len(records)} clips, {dict(counts)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
