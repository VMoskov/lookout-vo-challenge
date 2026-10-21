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
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import (DEFAULT_DATASET, SMOOTH_SIGMA_S, heading_constant_speed, load_manifest,
                    filter_flow_speed, load_pose_npz, load_timestamps, multi_run_heading_constant_speed,
                    multi_run_heading, _run_heading, heading_divergence_60s, integrate_planar, VGGT_SWITCH_DEG,
                    finite_fraction, total_turn_deg, prediction_dict, select_clips, validate_prediction, write_json)


def predict_clip(root, clip, pose_paths, sigma_s=SMOOTH_SIGMA_S, invert=False, speed_path=None,
                 speed_params=None, vggt_path=None, switch_deg=VGGT_SWITCH_DEG, vggt_member=False, flat_rule=None):
    """Returns (prediction dict, info dict) for one clip.

    `pose_paths` lists one npz per DPVO run (e.g. per seed); several runs are combined by the
    per-frame circular median of their headings. Missing or mismatched files are skipped.
    With `speed_path` (a flow_speed.py npz) the constant speed is replaced by the filtered flow
    speed (filter_flow_speed with `speed_params`); otherwise the speed is constant.
    With `vggt_path`, VGGT's heading replaces the median when it diverges from the DPVO-only
    median by > `switch_deg` within 60 s (None: no switch); `vggt_member` also adds VGGT as one
    more member of the median. `flat_rule` (X, Y, Z) also uses VGGT's heading when DPVO looks
    flat or broken: (DPVO median total turn < X deg or a seed's finite-pose fraction < Y) and
    VGGT's total turn > Z deg (None: no flat rule).
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
    dpvo_runs, vggt_run = list(runs), None
    if vggt_path is not None and vggt_path.is_file():
        zv = load_pose_npz(vggt_path)
        if not np.allclose(zv['times'], frame_times[zv['frame_indices']], rtol=0, atol=1e-6):
            raise ValueError(f'{vggt_path}: VGGT pose times do not match released frame timestamps')
        vggt_run = (zv['poses'], zv['times'])
        if vggt_member:
            runs.append(vggt_run)
    if invert:
        poses, times = runs[0] if runs else (None, None)
        positions, info = heading_constant_speed(poses, times, frame_times, sigma_s, invert=True)
    else:
        speed, speed_info = None, {}
        if speed_path is not None and speed_path.is_file():
            d = np.load(speed_path)
            if not np.allclose(d['times'], frame_times[d['frame_indices']], rtol=0, atol=1e-6):
                raise ValueError(f'{speed_path}: speed file frames do not match released timestamps')
            speed, speed_info = filter_flow_speed(d, frame_times, speed_params)
            if speed_info['speed_fallback'] != 'none':
                speed = None  # gate or reliability fallback: constant speed, exactly as without --speed
        elif speed_path is not None:
            speed_info = dict(speed_fallback='no speed file')
        positions, info = multi_run_heading_constant_speed(runs, frame_times, sigma_s, speed=speed)
        info.update(speed_info)
        hv = _run_heading(*vggt_run, frame_times, sigma_s)[0] if vggt_run is not None else None
        if hv is not None:
            h, _ = multi_run_heading(dpvo_runs, frame_times, sigma_s)
            use_vggt = False
            if switch_deg is not None and h is not None and info['method'] != 'straight_line_fallback':
                # DPVO/VGGT switch: VGGT's heading when it and the DPVO-only median diverge by > switch_deg within 60 s
                div = heading_divergence_60s(frame_times, h, hv)
                info.update(vggt_divergence_deg=div, vggt_switch=bool(div > switch_deg))
                use_vggt = div > switch_deg
            if flat_rule is not None:
                # flat-DPVO rule: DPVO barely turns or a seed lost most of its poses, while VGGT turns
                max_turn, min_finite, min_vggt_turn = flat_rule
                turn = total_turn_deg(h) if h is not None else 0.0
                finite = min([finite_fraction(p) for p, _ in dpvo_runs] + [0.0] * (len(pose_paths or []) - len(dpvo_runs)))
                vturn = total_turn_deg(hv)
                fires = (turn < max_turn or finite < min_finite) and vturn > min_vggt_turn
                info.update(dpvo_total_turn_deg=turn, min_finite_frac=finite, vggt_total_turn_deg=vturn,
                            flat_rule=bool(fires and not use_vggt))
                use_vggt = use_vggt or fires
            if use_vggt:
                positions = integrate_planar(frame_times, hv, speed if speed is not None else np.ones(len(frame_times)))
                info['method'] += '+vggt_heading'
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
    ap.add_argument('--speed', type=Path, help='flow_speed.py output dir: use flow speed instead of constant')
    ap.add_argument('--speed-params', type=Path, help='JSON of filter_flow_speed thresholds (fitted on train)')
    ap.add_argument('--speed-modes', default='wide', help='camera modes that use flow speed (others constant)')
    ap.add_argument('--vggt', type=Path, help='vggt_tool.py output dir: DPVO/VGGT heading switch')
    ap.add_argument('--switch-deg', type=float, default=VGGT_SWITCH_DEG)
    ap.add_argument('--switch-modes', default='wide,flir-VIS', help='camera modes where the switch may use VGGT')
    ap.add_argument('--vggt-member', action='store_true',
                    help='also add VGGT to the heading median (all camera modes); the switch still compares '
                         'the DPVO-only median with VGGT')
    ap.add_argument('--flat-rule', help='X,Y,Z: VGGT heading if (DPVO median total turn < X deg or a seed '
                                        'finite-pose fraction < Y) and VGGT total turn > Z deg')
    ap.add_argument('--flat-modes', default='wide,flir-VIS,flir-IR', help='camera modes where the flat rule applies')
    args = ap.parse_args(argv)
    if args.invert_poses and len(args.poses or []) > 1:
        ap.error('--invert-poses takes a single --poses directory')
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        ap.error('--output must be new or empty (or pass --overwrite)')
    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    root = args.dataset.resolve().parent
    speed_params = json.loads(args.speed_params.read_text()) if args.speed_params else None
    flat_rule = tuple(float(x) for x in args.flat_rule.split(',')) if args.flat_rule else None
    records = {}
    for clip in selected:
        key = clip['key']
        pose_paths = [d / 'poses' / f'{key}.npz' for d in args.poses or []]
        use_speed = args.speed is not None and clip['camera_mode'] in args.speed_modes.split(',')
        use_switch = args.vggt is not None and clip['camera_mode'] in args.switch_modes.split(',')
        use_flat = args.vggt is not None and flat_rule is not None and clip['camera_mode'] in args.flat_modes.split(',')
        use_vggt = use_switch or use_flat or (args.vggt is not None and args.vggt_member)
        pred, info = predict_clip(root, clip, pose_paths, args.sigma, args.invert_poses,
                                  args.speed / 'speed' / f'{key}.npz' if use_speed else None, speed_params,
                                  args.vggt / 'poses' / f'{key}.npz' if use_vggt else None,
                                  args.switch_deg if use_switch else None, args.vggt_member,
                                  flat_rule if use_flat else None)
        write_json(args.output / 'predictions' / f'{key}.json', pred)
        # JSON cannot hold NaN: a diagnostic that is undefined for a clip is recorded as null
        info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in info.items()}
        records[key] = dict(split=clip['split'], camera_mode=clip['camera_mode'],
                            n_frames=len(pred['times']), **info)
        print(f"{key}: {info['method']}" + (f" ({info['reason']})" if 'reason' in info else ''), flush=True)
    write_json(args.output / 'prediction_manifest.json',
               dict(poses=[str(d) for d in args.poses or []], sigma_s=args.sigma,
                    speed=str(args.speed) if args.speed else None, speed_params=speed_params,
                    speed_modes=args.speed_modes if args.speed else None,
                    vggt=str(args.vggt) if args.vggt else None, switch_deg=args.switch_deg if args.vggt else None,
                    switch_modes=args.switch_modes if args.vggt else None,
                    vggt_member=args.vggt_member if args.vggt else None,
                    flat_rule=list(flat_rule) if args.vggt and flat_rule else None,
                    flat_modes=args.flat_modes if args.vggt and flat_rule else None,
                    invert_poses=args.invert_poses, clips=records))
    counts = Counter(r['method'] for r in records.values())
    print(f'PREDICTIONS_COMPLETE: {len(records)} clips, {dict(counts)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
