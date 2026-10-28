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
                    multi_run_heading, _run_heading, circular_median, heading_divergence_60s, integrate_planar,
                    VGGT_SWITCH_DEG,
                    finite_fraction, total_turn_deg, pose_speed, apply_motion_priors, heading_change_rms_60s, POSE_SPEED_SIGMA_S, BROKEN_SEED_FINITE, BROKEN_SEED_TURN_DEG,
                    prediction_dict, select_clips, validate_prediction, write_json)


def pi3x_focal_scale(path):
    """Focal length a Pi3X / VGGT run estimates vs the nominal one (median window FOV): tan(nominal/2) / tan(est/2);
    > 1 = zoomed in. NaN without the run."""
    if not path.is_file():
        return float('nan')
    z = np.load(path)
    return float(np.tan(np.radians(float(z['nominal_hfov_deg'])) / 2) / np.tan(np.radians(float(np.median(z['window_hfov_deg']))) / 2))


def heading_from(path, frame_times, sigma_s):
    """Heading of one pose npz (e.g. VGGT/Pi3X) at every frame, or None if missing/unusable."""
    if path is None or not path.is_file():
        return None
    z = load_pose_npz(path)
    if not np.allclose(z['times'], frame_times[z['frame_indices']], rtol=0, atol=1e-6):
        raise ValueError(f'{path}: pose times do not match released frame timestamps')
    return _run_heading(z['poses'], z['times'], frame_times, sigma_s)[0]


def predict_clip(root, clip, pose_paths, sigma_s=SMOOTH_SIGMA_S, invert=False, speed_path=None,
                 speed_params=None, vggt_path=None, switch_deg=VGGT_SWITCH_DEG, vggt_member=False, flat_rule=None,
                 broken_seed=None, tiebreak_path=None, fm_paths=None, fm_combine=None, vggt_label='vggt',
                 fm_speed_path=None, fm_speed_mode=None, fm_speed_sigma=POSE_SPEED_SIGMA_S,
                 referee_path=None, referee_margin=0.0, referee_label='pi3x', agree_paths=None, agree_rule=None,
                 yaw_scale=1.0, vggt_sigma=None, max_yaw=None, max_accel=None, vggt_combine='median',
                 vggt_weights=None):
    """Returns (prediction dict, info dict) for one clip.

    `pose_paths` lists one npz per DPVO run (e.g. per seed); several runs are combined by the
    per-frame circular median of their headings. Missing or mismatched files are skipped.
    With `speed_path` (a flow_speed.py npz) the constant speed is replaced by the filtered flow
    speed (filter_flow_speed with `speed_params`); otherwise the speed is constant.
    With `vggt_path`, VGGT's heading replaces the median when it diverges from the DPVO-only
    median by > `switch_deg` within 60 s (None: no switch); `vggt_member` also adds VGGT as one
    more member of the median. `flat_rule` (X, Y, Z) also uses VGGT's heading when DPVO looks
    flat or broken: (DPVO median total turn < X deg or a seed's finite-pose fraction < Y) and
    VGGT's total turn > Z deg (None: no flat rule). `broken_seed` (F, T) uses VGGT's heading when a
    seed's finite-pose fraction < F and the DPVO median total turn < T deg (None: off).
    With `tiebreak_path` (a third heading source, e.g. Pi3X, same npz format) a firing switch is
    kept only if VGGT's heading is closer to it than the DPVO median's (60 s divergence).
    `fm_paths` are further foundation-model heading runs (e.g. Pi3X, same npz format), used as
    `fm_combine`: 'members' (extra members of the per-frame median), 'median3' (circular median of
    the DPVO median and each of them; VGGT's run joins via fm_paths too) or 'blend' (when VGGT's
    heading replaces the median, use the circular mean of VGGT and them instead).
    `vggt_path` may be a list of runs (e.g. Pi3X at several keyframe phases): their per-frame
    circular median is the heading. `fm_speed_path` (a foundation-model pose npz) gives a speed from
    its camera centres (pose_speed): 'replace' uses it instead of flow/constant speed, 'fallback'
    only where flow is gated off or missing. With `referee_path` (e.g. Pi3X) VGGT referees between
    the DPVO median and that model: the DPVO median is kept (with the switch/rules) if VGGT's 60 s
    heading changes are closer to it (heading_change_rms_60s) by more than `referee_margin` deg,
    else the referee model's heading is used. `agree_rule` (T, D) with `agree_paths` (two foundation-model
    runs, e.g. VGGT and Pi3X): if the DPVO median turns < T deg in total and the two agree (60 s
    heading-change RMS < D deg), the circular mean of their headings is used. `yaw_scale` multiplies
    the heading change when the --vggt source's heading (after any blend) replaces the median
    (rotation-scale calibration). `vggt_weights` (one per --vggt run) weight its circular mean
    (e.g. Pi3X 0.8 / VGGT 0.2; missing and zero-weight runs are left out, equal weights = the plain mean). `vggt_sigma` smooths the --vggt source (default `sigma_s`).
    `max_yaw` (deg/s) / `max_accel` (x median speed per s) apply physical priors to the final path
    (common.apply_motion_priors).
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
    dpvo_runs, vggt_runs, vggt_w = list(runs), [], []
    vps = [vggt_path] if isinstance(vggt_path, Path) else vggt_path or []
    for vp, w in zip(vps, vggt_weights or [1.0] * len(vps)):
        if vp.is_file() and w != 0:
            zv = load_pose_npz(vp)
            if not np.allclose(zv['times'], frame_times[zv['frame_indices']], rtol=0, atol=1e-6):
                raise ValueError(f'{vp}: VGGT pose times do not match released frame timestamps')
            vggt_runs.append((zv['poses'], zv['times']))
            vggt_w.append(w)
    if vggt_member:
        runs += vggt_runs
    fm_runs = []
    for path in fm_paths or []:
        if path.is_file():
            zf = load_pose_npz(path)
            if not np.allclose(zf['times'], frame_times[zf['frame_indices']], rtol=0, atol=1e-6):
                raise ValueError(f'{path}: pose times do not match released frame timestamps')
            fm_runs.append((zf['poses'], zf['times']))
    if fm_combine == 'members':
        runs += fm_runs
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
        if fm_speed_path is not None and fm_speed_path.is_file() and (fm_speed_mode == 'replace' or speed is None):
            zs = load_pose_npz(fm_speed_path)
            fs = pose_speed(zs['poses'], zs['times'], frame_times, fm_speed_sigma, disp=zs.get('window_disp'))
            if fs is not None:
                speed = fs
                speed_info['speed_source'] = 'pose_speed'
        positions, info = multi_run_heading_constant_speed(runs, frame_times, sigma_s, speed=speed)
        final_h, final_speed = (multi_run_heading(runs, frame_times, sigma_s)[0]   # DPVO-only path
                                if info['method'] != 'straight_line_fallback' else None), speed
        info.update(speed_info)
        fm_heads = [x for x in (_run_heading(*r, frame_times, sigma_s)[0] for r in fm_runs) if x is not None]
        if fm_combine == 'median3' and fm_heads and info['method'] != 'straight_line_fallback':
            # per-frame circular median of the DPVO median and each foundation model
            h3 = circular_median([multi_run_heading(dpvo_runs, frame_times, sigma_s)[0]] + fm_heads)
            positions = integrate_planar(frame_times, h3, speed if speed is not None else np.ones(len(frame_times)))
            final_h, final_speed = h3, speed
            info['method'] += f'+median_of_{1 + len(fm_heads)}'
        vs = vggt_sigma if vggt_sigma is not None else sigma_s
        if vggt_combine == 'mean' and len(vggt_runs) > 1:
            hw = [(x, w) for x, w in zip((_run_heading(*r, frame_times, vs)[0] for r in vggt_runs), vggt_w) if x is not None]
            hs = [x for x, _ in hw]
            if hs and len({w for _, w in hw}) > 1:
                hv = hs[0] + np.angle(np.average([np.exp(1j * (x - hs[0])) for x in hs], axis=0, weights=[w for _, w in hw]))
            else:
                hv = hs[0] + np.angle(np.mean([np.exp(1j * (x - hs[0])) for x in hs], axis=0)) if hs else None
        else:
            hv = multi_run_heading(vggt_runs, frame_times, vs)[0] if vggt_runs else None
        if hv is not None:
            h, _ = multi_run_heading(dpvo_runs, frame_times, sigma_s)
            use_vggt = False
            if referee_path is not None and referee_path.is_file() and h is not None:
                zr = load_pose_npz(referee_path)
                hr = _run_heading(zr['poses'], zr['times'], frame_times, sigma_s)[0]
                if hr is not None:
                    d_dpvo, d_ref = heading_change_rms_60s(frame_times, h, hv), heading_change_rms_60s(frame_times, hr, hv)
                    use_ref = not d_dpvo + referee_margin < d_ref
                    info.update(referee_dpvo_deg=d_dpvo, referee_fm_deg=d_ref, referee_uses_fm=bool(use_ref))
                    if use_ref:
                        positions = integrate_planar(frame_times, hr, speed if speed is not None else np.ones(len(frame_times)))
                        final_h, final_speed = hr, speed
                        info['method'] += f'+{referee_label}_referee'
                        switch_deg = broken_seed = flat_rule = None   # the referee model's heading is final
            if switch_deg is not None and h is not None and info['method'] != 'straight_line_fallback':
                # DPVO/VGGT switch: VGGT's heading when it and the DPVO-only median diverge by > switch_deg within 60 s
                div = heading_divergence_60s(frame_times, h, hv)
                info.update(vggt_divergence_deg=div, vggt_switch=bool(div > switch_deg))
                use_vggt = div > switch_deg
                if use_vggt and tiebreak_path is not None and tiebreak_path.is_file():
                    zt = load_pose_npz(tiebreak_path)
                    ht = _run_heading(zt['poses'], zt['times'], frame_times, sigma_s)[0]
                    if ht is not None:
                        d_dpvo, d_vggt = heading_divergence_60s(frame_times, h, ht), heading_divergence_60s(frame_times, hv, ht)
                        use_vggt = d_vggt < d_dpvo
                        info.update(tiebreak_dpvo_deg=d_dpvo, tiebreak_vggt_deg=d_vggt, vggt_switch=bool(use_vggt))
            turn = total_turn_deg(h) if h is not None else 0.0
            finite = min([finite_fraction(p) for p, _ in dpvo_runs] + [0.0] * (len(pose_paths or []) - len(dpvo_runs)))
            if broken_seed is not None:
                # broken-seed rule: a seed lost most of its poses and the DPVO median barely turns
                min_finite, max_turn = broken_seed
                fires = finite < min_finite and turn < max_turn
                info.update(dpvo_total_turn_deg=turn, min_finite_frac=finite, broken_seed_rule=bool(fires and not use_vggt))
                use_vggt = use_vggt or fires
            if flat_rule is not None:
                # flat-DPVO rule: DPVO barely turns or a seed lost most of its poses, while VGGT turns
                max_turn, min_finite, min_vggt_turn = flat_rule
                vturn = total_turn_deg(hv)
                fires = (turn < max_turn or finite < min_finite) and vturn > min_vggt_turn
                info.update(dpvo_total_turn_deg=turn, min_finite_frac=finite, vggt_total_turn_deg=vturn,
                            flat_rule=bool(fires and not use_vggt))
                use_vggt = use_vggt or fires
            if use_vggt:
                if fm_combine == 'blend' and fm_heads:
                    # circular mean of VGGT and the other foundation models, unwrapped along VGGT
                    hv = hv + np.angle(np.mean([np.exp(1j * (x - hv)) for x in [hv] + fm_heads], axis=0))
                    info['method'] += '+fm_blend'
                if yaw_scale != 1.0:
                    hv = yaw_scale * hv          # rotation-scale calibration of the final heading
                positions = integrate_planar(frame_times, hv, speed if speed is not None else np.ones(len(frame_times)))
                final_h, final_speed = hv, speed
                info['method'] += f'+{vggt_label}_heading'
        if agree_rule is not None and agree_paths and info['method'] != 'straight_line_fallback':
            ha, hb = (heading_from(p, frame_times, sigma_s) for p in agree_paths)
            h = multi_run_heading(dpvo_runs, frame_times, sigma_s)[0]
            if ha is not None and hb is not None and h is not None:
                turn, rms = total_turn_deg(h), heading_change_rms_60s(frame_times, ha, hb)
                fires = turn < agree_rule[0] and rms < agree_rule[1]
                info.update(agree_dpvo_turn_deg=turn, agree_rms_deg=rms, agree_rule=bool(fires))
                if fires:
                    hm = ha + np.angle(0.5 * (1 + np.exp(1j * (hb - ha))))   # circular mean, unwrapped along ha
                    positions = integrate_planar(frame_times, hm, speed if speed is not None else np.ones(len(frame_times)))
                    final_h, final_speed = hm, speed
                    info['method'] += '+fm_agree_heading'
    if errors:
        info['pose_errors'] = errors
    if not runs:
        info['reason'] = 'no usable pose file'
    if max_yaw is not None or max_accel is not None:
        positions = apply_motion_priors(frame_times, positions, max_yaw, max_accel)
        info.update(max_yaw_deg_s=max_yaw, max_accel=max_accel)
    pred = prediction_dict(clip['key'], frame_times, positions)
    validate_prediction(pred, clip['key'], frame_times)
    if not invert and max_yaw is None and max_accel is None and final_h is not None:
        info['_final'] = (final_h, final_speed)   # popped by main (--save-heading)
    return pred, info


def per_mode(spec, mode):
    """'X' or 'MODE=X,MODE=X' -> float for this camera mode, or None (no limit / mode not listed / 'inf')."""
    if not spec:
        return None
    vals = {'*': spec} if '=' not in spec else dict(kv.split('=') for kv in spec.split(','))
    v = vals.get(mode, vals.get('*'))
    return None if v is None or v == 'inf' else float(v)


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
    ap.add_argument('--save-heading', action='store_true',
                    help='also write <output>/heading/<clip>.npz: the final heading and speed of the path (oracle speed fits)')
    ap.add_argument('--speed', type=Path, help='flow_speed.py output dir: use flow speed instead of constant')
    ap.add_argument('--speed-params', type=Path, help='JSON of filter_flow_speed thresholds (fitted on train)')
    ap.add_argument('--speed-modes', default='wide', help='camera modes that use flow speed (others constant)')
    ap.add_argument('--vggt', type=Path, nargs='+',
                    help='vggt_tool.py / pi3_tool.py output dir(s): DPVO/VGGT heading switch; several = their median')
    ap.add_argument('--vggt-combine', choices=['median', 'mean'], default='median',
                    help='several --vggt runs: per-frame circular median (default) or circular mean')
    ap.add_argument('--vggt-weights', nargs='+',
                    help='W,W,... (one per --vggt run) or MODE=W,W,... per camera mode, for --vggt-combine mean; '
                         'a weight of 0 leaves that run out')
    ap.add_argument('--vggt-label', default='vggt', help='name of the --vggt source in the method label')
    ap.add_argument('--fm-speed', type=Path, help='foundation-model run dir: speed from its camera centres')
    ap.add_argument('--fm-speed-mode', choices=['replace', 'fallback'], default='replace')
    ap.add_argument('--fm-speed-sigma', type=float, default=POSE_SPEED_SIGMA_S, help='pose speed smoothing sigma, s')
    ap.add_argument('--switch-deg', type=float, default=VGGT_SWITCH_DEG)
    ap.add_argument('--switch-modes', default='wide,flir-VIS', help='camera modes where the switch may use VGGT')
    ap.add_argument('--vggt-member', action='store_true',
                    help='also add VGGT to the heading median (all camera modes); the switch still compares '
                         'the DPVO-only median with VGGT')
    ap.add_argument('--flat-rule', help='X,Y,Z: VGGT heading if (DPVO median total turn < X deg or a seed '
                                        'finite-pose fraction < Y) and VGGT total turn > Z deg')
    ap.add_argument('--referee', type=Path, help='run dir (e.g. Pi3X): VGGT referees between the DPVO median and it')
    ap.add_argument('--referee-margin', type=float, default=0.0, help='deg by which VGGT must favour DPVO to keep it')
    ap.add_argument('--referee-label', default='pi3x')
    ap.add_argument('--agree-rule', help='T,D: if the DPVO median turns < T deg and --agree-a/--agree-b agree '
                                         '(60 s RMS < D deg), use their circular mean heading')
    ap.add_argument('--agree-a', type=Path, help='first run dir for --agree-rule (e.g. VGGT)')
    ap.add_argument('--agree-b', type=Path, help='second run dir for --agree-rule (e.g. Pi3X)')
    ap.add_argument('--agree-modes', default='flir-IR', help='camera modes where --agree-rule applies')
    ap.add_argument('--vggt-sigma', type=float, help='heading smoothing sigma (s) for the --vggt source (default --sigma)')
    ap.add_argument('--max-yaw-rate', help='R or MODE=R,...: limit the path heading rate to R deg/s (physical prior)')
    ap.add_argument('--max-accel', help='A or MODE=A,...: limit speed change to A x median speed per s (physical prior)')
    ap.add_argument('--zoom-yaw-scale', help='MODE=K,T: on clips of MODE whose first --vggt run (Pi3X) estimates a focal length > T x '
                                              'the nominal one (zoomed in), multiply that mode\'s --yaw-scale by K')
    ap.add_argument('--yaw-scale', default='1.0',
                    help='K or MODE=K,MODE=K: multiply the --vggt heading change by K (per camera mode)')
    ap.add_argument('--tiebreak', type=Path, help='third heading source run dir (e.g. Pi3X): decides a firing switch')
    ap.add_argument('--fm', type=Path, nargs='+', help='further foundation-model run dirs (e.g. Pi3X) for --fm-combine')
    ap.add_argument('--fm-combine', choices=['members', 'median3', 'blend'])
    ap.add_argument('--fm-modes', default='wide,flir-VIS', help='camera modes where --fm is used')
    ap.add_argument('--broken-seed', default=f'{BROKEN_SEED_FINITE},{BROKEN_SEED_TURN_DEG}',
                    help="F,T: in --switch-modes, VGGT heading if a seed's finite-pose fraction < F and the DPVO "
                         "median total turn < T deg (default on; 'off' disables)")
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
    broken_seed = None if args.broken_seed == 'off' else tuple(float(x) for x in args.broken_seed.split(','))
    yaw_scales = ({'*': float(args.yaw_scale)} if '=' not in args.yaw_scale else
                  {kv.split('=')[0]: float(kv.split('=')[1]) for kv in args.yaw_scale.split(',')})
    agree_rule = tuple(float(x) for x in args.agree_rule.split(',')) if args.agree_rule else None
    if agree_rule and not (args.agree_a and args.agree_b):
        ap.error('--agree-rule needs --agree-a and --agree-b')
    flat_rule = tuple(float(x) for x in args.flat_rule.split(',')) if args.flat_rule else None
    vggt_weights = {(s.split('=')[0] if '=' in s else '*'): [float(x) for x in s.split('=')[-1].split(',')]
                    for s in args.vggt_weights or []}
    if vggt_weights and (args.vggt_combine != 'mean' or any(len(w) != len(args.vggt or []) for w in vggt_weights.values())):
        ap.error('--vggt-weights needs --vggt-combine mean and one weight per --vggt run')
    zoom = None
    if args.zoom_yaw_scale:
        zm, zv = args.zoom_yaw_scale.split('=')
        zoom = (zm, *(float(x) for x in zv.split(',')))
        if not args.vggt:
            ap.error('--zoom-yaw-scale needs --vggt (the first run gives the focal estimate)')
    records = {}
    for clip in selected:
        key = clip['key']
        ys = yaw_scales.get(clip['camera_mode'], yaw_scales.get('*', 1.0))
        zoom_info = {}
        if zoom is not None and clip['camera_mode'] == zoom[0]:
            fs = pi3x_focal_scale(args.vggt[0] / 'poses' / f'{key}.npz')
            zoom_info = dict(zoom_focal_scale=fs, zoom_yaw_scale=zoom[1] if fs > zoom[2] else 1.0)
            ys *= zoom_info['zoom_yaw_scale']
        pose_paths = [d / 'poses' / f'{key}.npz' for d in args.poses or []]
        use_speed = args.speed is not None and clip['camera_mode'] in args.speed_modes.split(',')
        use_switch = args.vggt is not None and clip['camera_mode'] in args.switch_modes.split(',')
        use_flat = args.vggt is not None and flat_rule is not None and clip['camera_mode'] in args.flat_modes.split(',')
        use_vggt = use_switch or use_flat or (args.vggt is not None and args.vggt_member)
        pred, info = predict_clip(root, clip, pose_paths, args.sigma, args.invert_poses,
                                  args.speed / 'speed' / f'{key}.npz' if use_speed else None, speed_params,
                                  [d / 'poses' / f'{key}.npz' for d in args.vggt] if use_vggt else None,
                                  args.switch_deg if use_switch else None, args.vggt_member,
                                  flat_rule if use_flat else None, broken_seed if use_switch else None,
                                  args.tiebreak / 'poses' / f'{key}.npz' if use_switch and args.tiebreak else None,
                                  [d / 'poses' / f'{key}.npz' for d in args.fm or []]
                                  if clip['camera_mode'] in args.fm_modes.split(',') else None, args.fm_combine,
                                  args.vggt_label, args.fm_speed / 'poses' / f'{key}.npz' if args.fm_speed and use_speed
                                  else None, args.fm_speed_mode, args.fm_speed_sigma,
                                  args.referee / 'poses' / f'{key}.npz' if use_switch and args.referee else None,
                                  args.referee_margin, args.referee_label,
                                  [d / 'poses' / f'{key}.npz' for d in (args.agree_a, args.agree_b)]
                                  if agree_rule and clip['camera_mode'] in args.agree_modes.split(',') else None,
                                  agree_rule, ys,
                                  args.vggt_sigma, per_mode(args.max_yaw_rate, clip['camera_mode']),
                                  per_mode(args.max_accel, clip['camera_mode']), args.vggt_combine,
                                  vggt_weights.get(clip['camera_mode'], vggt_weights.get('*')))
        info.update(zoom_info)
        write_json(args.output / 'predictions' / f'{key}.json', pred)
        final = info.pop('_final', None)
        if args.save_heading and final is not None:
            (args.output / 'heading').mkdir(parents=True, exist_ok=True)
            np.savez(args.output / 'heading' / f'{key}.npz', heading=final[0],
                     speed=final[1] if final[1] is not None else np.ones(len(pred['times'])), speed_from_flow=final[1] is not None)
        # JSON cannot hold NaN: a diagnostic that is undefined for a clip is recorded as null
        info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in info.items()}
        records[key] = dict(split=clip['split'], camera_mode=clip['camera_mode'],
                            n_frames=len(pred['times']), **info)
        print(f"{key}: {info['method']}" + (f" ({info['reason']})" if 'reason' in info else ''), flush=True)
    write_json(args.output / 'prediction_manifest.json',
               dict(poses=[str(d) for d in args.poses or []], sigma_s=args.sigma,
                    speed=str(args.speed) if args.speed else None, speed_params=speed_params,
                    speed_modes=args.speed_modes if args.speed else None,
                    vggt=[str(d) for d in args.vggt] if args.vggt else None, vggt_label=args.vggt_label if args.vggt else None,
                    fm_speed=str(args.fm_speed) if args.fm_speed else None,
                    referee=str(args.referee) if args.referee else None,
                    agree_rule=list(agree_rule) if agree_rule else None,
                    yaw_scale=yaw_scales if args.vggt else None, zoom_yaw_scale=args.zoom_yaw_scale,
                    vggt_combine=args.vggt_combine if args.vggt and len(args.vggt) > 1 else None,
                    vggt_weights=vggt_weights or None,
                    max_yaw_rate=args.max_yaw_rate, max_accel=args.max_accel,
                    vggt_sigma=args.vggt_sigma if args.vggt else None,
                    agree=[str(args.agree_a), str(args.agree_b)] if agree_rule else None,
                    agree_modes=args.agree_modes if agree_rule else None,
                    referee_margin=args.referee_margin if args.referee else None,
                    fm_speed_mode=args.fm_speed_mode if args.fm_speed else None,
                    fm_speed_sigma=args.fm_speed_sigma if args.fm_speed else None, switch_deg=args.switch_deg if args.vggt else None,
                    switch_modes=args.switch_modes if args.vggt else None,
                    vggt_member=args.vggt_member if args.vggt else None,
                    broken_seed=list(broken_seed) if args.vggt and broken_seed else None,
                    tiebreak=str(args.tiebreak) if args.vggt and args.tiebreak else None,
                    fm=[str(d) for d in args.fm] if args.fm else None, fm_combine=args.fm_combine,
                    fm_modes=args.fm_modes if args.fm else None,
                    flat_rule=list(flat_rule) if args.vggt and flat_rule else None,
                    flat_modes=args.flat_modes if args.vggt and flat_rule else None,
                    invert_poses=args.invert_poses, clips=records))
    counts = Counter(r['method'] for r in records.values())
    print(f'PREDICTIONS_COMPLETE: {len(records)} clips, {dict(counts)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
