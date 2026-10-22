#!/usr/bin/env python3
"""Oracle analysis on train/val. This is the ONLY fermarble code that reads references.

It derives the true heading and true speed from each reference and scores these variants
with the starter metrics (metrics.score_trajectory, the evaluator's own scoring call):

  straight              constant-velocity straight line
  raw_dpvo              DPVO positions as written by the plain baseline
  dpvo_head_const       DPVO camera heading + constant speed (= make_predictions.py)
  dpvo_head_inv_const   same with every pose inverted (pose-convention check)
  dpvo_motion_const     direction of DPVO's own translation + constant speed
  dpvo_head_true_speed  DPVO camera heading + true speed
  true_head_const       true heading + constant speed
  true_head_dpvo_speed  true heading + DPVO translation speed
  true_head_true_speed  true heading + true speed (pipeline sanity check: must be ~0)

Clips without usable poses use the submission fallback (a straight line) for dpvo_head_const and
dpvo_head_inv_const. They are NaN for the other DPVO variants and are left out of those means.
Optionally scores a real predictions directory with starter evaluate.evaluate_clip.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

from common import (DEFAULT_DATASET, NO_LIMIT, REPO_ROOT, SMOOTH_SIGMA_S, START_WINDOW_S, clean_poses,
                    filter_flow_speed, flow_reliable_speed, flow_speed_from_reliable, multi_run_heading,
                    read_json, rolling_median, speed_variation, flow_quality_features, horizon_jump,
                    quat_to_rotmat, estimate_up, forward_motion_cosine, heading_divergence_60s, gap_aware_coverage,
                    forward_motion_cosine, heading_at_frames, heading_constant_speed, integrate_planar, load_manifest,
                    load_pose_npz, load_timestamps, motion_heading, pose_heading, select_clips,
                    smooth_in_time, straight_line, write_json)
import metrics  # starter
from evaluate import evaluate_clip  # starter
from public_dataset import load_reference  # starter

VARIANTS = ['straight', 'raw_dpvo', 'dpvo_head_const', 'dpvo_head_inv_const', 'dpvo_motion_const',
            'dpvo_head_true_speed', 'true_head_const', 'true_head_dpvo_speed', 'true_head_true_speed']
STATIONARY_MPS = 0.05


def true_motion(ref_t, ref_xyz):
    """Per-reference-interval heading (unwrapped compass bearing, rad, clockwise from north) and
    speed (m/s) of the planar reference, whose x/y are east/north."""
    vel = np.diff(ref_xyz[:, :2], axis=0) / np.diff(ref_t)[:, None]  # (east, north)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 0], vel[:, 1])  # compass bearing: clockwise from north
    moving = speed > STATIONARY_MPS
    idx = np.arange(len(speed))
    if moving.sum() >= 1:  # stationary intervals have no heading: carry it from neighbours
        heading = np.interp(idx, idx[moving], np.unwrap(heading[moving]))
    return np.unwrap(heading), speed


def at_frames_piecewise(ref_t, per_interval, frame_times):
    """Evaluate a per-reference-interval quantity at frame times (constant within an interval)."""
    k = np.clip(np.searchsorted(ref_t, frame_times, side='right') - 1, 0, len(per_interval) - 1)
    return per_interval[k]


def score(est_t, est_xyz, ref_t, ref_xyz):
    """Drift % exactly as starter evaluate.py computes it, or NaN when not scoreable."""
    est_xyz = np.asarray(est_xyz, float)
    if not np.isfinite(est_xyz).all() or len(est_t) < 2:
        return float('nan')
    vo = metrics.score_trajectory(np.asarray(est_t, float), est_xyz, ref_t, ref_xyz)
    return float(vo['drift_pct']) if vo is not None and np.isfinite(vo['drift_pct']) else float('nan')


def speed_stats(ref_t, per_interval_speed, sigma_s=SMOOTH_SIGMA_S):
    """How far the true speed is from constant: CV (std/mean) and max/median of the speed,
    Gaussian-smoothed like the headings (sigma_s) so single-sample spikes do not dominate."""
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    _, v = smooth_in_time(mid, per_interval_speed, sigma_s)
    med = float(np.median(v))
    return dict(mean_speed_mps=float(v.mean()),
                speed_cv=float(v.std() / v.mean()) if v.mean() > 0 else float('nan'),
                speed_max_median=float(v.max() / med) if med > 0 else float('nan'))


GROSS_HEADING_DEG = 90.0  # |DPVO - true heading| beyond this while moving = tracking failure


def classify_clip(row, hd_series=None):
    """Which ingredient a clip's error comes from: 'dpvo_failure' (NaN poses, or the DPVO heading
    off by >= GROSS_HEADING_DEG while moving), else 'speed' or 'heading' by which oracle swap
    (true speed or true heading) removes more of our drift."""
    nan_poses = row.get('n_finite_poses', 0) < row.get('n_poses', 0)
    gross = False
    if hd_series is not None and hd_series['moving'].any():
        gross = bool(np.abs(hd_series['diff'][hd_series['moving']]).max() >= GROSS_HEADING_DEG)
    speed_gain = row['dpvo_head_const'] - row['dpvo_head_true_speed']
    heading_gain = row['dpvo_head_const'] - row['true_head_const']
    if nan_poses or gross:
        cat = 'dpvo_failure'
    else:
        cat = 'speed' if speed_gain >= heading_gain else 'heading'
    return dict(category=cat, nan_poses=nan_poses, gross_heading_error=gross,
                speed_gain=speed_gain, heading_gain=heading_gain)


# Fixing a category = swapping in the truth for the ingredient that category blames.
FIXED_VARIANT = dict(speed='dpvo_head_true_speed', heading='true_head_const', dpvo_failure='true_head_const')


def worst_report(rows, hd_series, n_worst):
    """Table of the n worst clips by our drift, with category, and the mean drop per category fix."""
    for r in rows:
        r.update(classify_clip(r, hd_series.get(r['key'])))
    worst = sorted(rows, key=lambda r: -r['dpvo_head_const'])[:n_worst]
    lines = [f'== {n_worst} worst clips by our drift (DPVO heading + constant speed)',
             f"{'clip':<9}{'camera':<10}{'ours':>7}{'dH+tV':>7}{'tH+cV':>7}  {'NaN':<4}{'spd CV':>7}"
             f"{'max/med':>8}  category"]
    for r in worst:
        lines.append(f"{r['key']:<9}{r['camera_mode']:<10}{r['dpvo_head_const']:>7.1f}"
                     f"{r['dpvo_head_true_speed']:>7.1f}{r['true_head_const']:>7.1f}  "
                     f"{'yes' if r['nan_poses'] else 'no':<4}{r['speed_cv']:>7.2f}{r['speed_max_median']:>8.2f}  "
                     f"{r['category']}")
    mean = np.mean([r['dpvo_head_const'] for r in rows])
    lines.append(f'\nmean drift now {mean:.2f}% over {len(rows)} clips. Estimated mean if a category were fixed'
                 ' (its clips get the oracle swap; categories are not additive):')
    for scope, sel in ((f'worst {n_worst} only', worst), (f'all {len(rows)} clips', rows)):
        for cat, variant in FIXED_VARIANT.items():
            clips = [r for r in sel if r['category'] == cat]
            drop = sum(r['dpvo_head_const'] - r[variant] for r in clips) / len(rows)
            lines.append(f'  {scope:<14} {cat:<13} n={len(clips):<3} fix -> {mean - drop:6.2f}%  (drop {drop:5.2f})')
    return '\n'.join(lines)


def analyse_clip(root, clip, pose_path, sigma_s=SMOOTH_SIGMA_S):
    frame_t = load_timestamps(root, clip)
    ref_t, ref_xyz = load_reference(root, clip)
    true_h, true_v = true_motion(ref_t, ref_xyz)
    th = at_frames_piecewise(ref_t, true_h, frame_t)
    tv = at_frames_piecewise(ref_t, true_v, frame_t)
    ones = np.ones(len(frame_t))
    nan = float('nan')
    row = dict(key=clip['key'], split=clip['split'], camera_mode=clip['camera_mode'],
               path_length_m=float(metrics.cumulative_length(ref_xyz)[-1]),
               ref_points=len(ref_t), n_poses=0, method='no_pose_file')
    row.update(speed_stats(ref_t, true_v, sigma_s))

    def s(xyz, t=frame_t):
        return score(t, xyz, ref_t, ref_xyz)

    row['straight'] = s(straight_line(frame_t))
    row['true_head_const'] = s(integrate_planar(frame_t, th, ones))
    row['true_head_true_speed'] = s(integrate_planar(frame_t, th, tv))

    raw = None
    if pose_path is not None and pose_path.is_file():
        raw = load_pose_npz(pose_path)
    pipe, info = heading_constant_speed(raw['poses'] if raw else None, raw['times'] if raw else None,
                                        frame_t, sigma_s)
    inv, _ = heading_constant_speed(raw['poses'] if raw else None, raw['times'] if raw else None,
                                    frame_t, sigma_s, invert=True)
    row['dpvo_head_const'] = s(pipe)
    row['dpvo_head_inv_const'] = s(inv)
    row['method'] = info['method'] if raw else 'no_pose_file'
    row['pipeline_gap_cov'] = gap_aware_coverage(frame_t, ref_t)
    for v in ('raw_dpvo', 'dpvo_motion_const', 'dpvo_head_true_speed', 'true_head_dpvo_speed'):
        row[v] = nan
    if raw is not None:
        row['n_poses'] = int(len(raw['poses']))
        row['n_finite_poses'] = int(np.isfinite(raw['poses']).all(1).sum())
        row['raw_dpvo'] = s(raw['poses'][:, :3], raw['times'])
        row['raw_overlap_cov'] = float(metrics.overlap_window(raw['times'], ref_t).mean())
        poses, pt = clean_poses(raw['poses'], raw['times'])
        if poses is not None:
            row['fwd_cos'] = forward_motion_cosine(poses, pt, sigma_s)
            row['fwd_cos_inv'] = forward_motion_cosine(poses, pt, sigma_s, invert=True)
            heading, _ = pose_heading(poses, pt)
            if heading is not None:
                dh = heading_at_frames(pt, heading, frame_t, sigma_s)
                row['dpvo_head_true_speed'] = s(integrate_planar(frame_t, dh, tv))
            mh = motion_heading(poses, pt, sigma_s)
            if mh is not None:
                grid, mhead, mspeed = mh
                row['dpvo_motion_const'] = s(integrate_planar(frame_t, np.interp(frame_t, grid, mhead), ones))
                row['true_head_dpvo_speed'] = s(integrate_planar(frame_t, th, np.interp(frame_t, grid, mspeed)))
    return row


# --------------------------------------------------------------------------------------
# DPVO heading vs true heading over time
# --------------------------------------------------------------------------------------
HEADING_GRID_HZ = 2.0
MIN_SPEED_MPS = 1.0  # below this the direction of travel is too noisy to compare with the bow


def _lstsq(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, X @ coef, float(np.sqrt(np.mean(resid ** 2)))


def heading_difference(root, clip, pose_path, sigma_s=SMOOTH_SIGMA_S):
    """Compare DPVO camera heading with the reference direction of travel over time.

    Both are smoothed identically (Gaussian, sigma_s), put on a common 2 Hz grid over their
    shared window and expressed as "turned since the start" (start = first START_WINDOW_S s).
    The difference d(t) = DPVO turned - true turned (deg, clockwise positive) is then explained
    on moving samples (true speed >= MIN_SPEED_MPS) by:
      drift    d = a + b t                         (DPVO yaw drift: grows with time)
      current  d = a + (-Ue cos th + Un sin th)/v  (bow vs travel direction under a constant
                                                    water current U, th = true bearing, v = speed)
      both     drift + current
    Returns (summary dict, series dict) or None when there are no usable poses.
    """
    if pose_path is None or not pose_path.is_file():
        return None
    raw = load_pose_npz(pose_path)
    poses, pt = clean_poses(raw['poses'], raw['times'])
    if poses is None:
        return None
    dh, _ = pose_heading(poses, pt)
    if dh is None:
        return None
    ref_t, ref_xyz = load_reference(root, clip)
    tb, tv = true_motion(ref_t, ref_xyz)
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    t0, t1 = max(pt[0], mid[0]), min(pt[-1], mid[-1])
    if t1 - t0 < 30:
        return None
    grid = np.arange(t0, t1, 1 / HEADING_GRID_HZ)
    smooth = lambda t, v: np.interp(grid, *smooth_in_time(t, v, sigma_s))
    dpvo, true, speed = smooth(pt, dh), smooth(mid, tb), smooth(mid, tv)
    start = grid <= grid[0] + START_WINDOW_S
    dpvo_turned = np.degrees(dpvo - dpvo[start].mean())
    true_turned = np.degrees(true - true[start].mean())
    # Wrapped to (-180, 180], NOT unwrapped: one momentary glitch above 180 deg would otherwise add
    # a permanent full turn to everything after it and fake a huge drift.
    diff = np.degrees(np.angle(np.exp(1j * np.radians(dpvo_turned - true_turned))))
    moving = speed >= MIN_SPEED_MPS
    tm = (grid - grid[0]) / 60.0  # minutes since start
    row = dict(key=clip['key'], split=clip['split'], camera_mode=clip['camera_mode'],
               duration_min=float(tm[-1]), moving_frac=float(moving.mean()),
               end_diff_deg=float(diff[grid >= grid[-1] - 10].mean()),
               max_abs_diff_deg=float(np.abs(diff).max()))
    fits = {}
    if moving.sum() >= 20:
        y = diff[moving]
        ones = np.ones(moving.sum())
        cur = np.column_stack([np.cos(true[moving]), np.sin(true[moving])]) / speed[moving, None]
        X = dict(constant=ones[:, None], drift=np.column_stack([ones, tm[moving]]),
                 current=np.column_stack([ones, np.degrees(cur)]),
                 both=np.column_stack([ones, tm[moving], np.degrees(cur)]))
        rmse = {}
        for name, Xi in X.items():
            coef, fitted, rmse[name] = _lstsq(Xi, y)
            full = np.full(len(grid), np.nan)
            full[moving] = fitted
            fits[name] = (coef, full)
        var0 = rmse['constant'] ** 2
        r2 = {k: (1 - rmse[k] ** 2 / var0) if var0 > 0 else 0.0 for k in ('drift', 'current', 'both')}
        b = fits['drift'][0][1]
        ue, un = -fits['current'][0][1], fits['current'][0][2]  # regressors in deg: coef is m/s
        row.update(diff_std_deg=rmse['constant'], drift_deg_per_min=float(b),
                   drift_over_clip_deg=float(b * tm[-1]), r2_drift=r2['drift'], r2_current=r2['current'],
                   r2_both=r2['both'], current_east_mps=float(ue), current_north_mps=float(un),
                   current_mps=float(np.hypot(ue, un)), rmse_both_deg=rmse['both'])
        row['label'] = classify_difference(row)
    else:
        row['label'] = 'too_slow'
    series = dict(t=grid - grid[0], dpvo_turned=dpvo_turned, true_turned=true_turned, diff=diff,
                  speed=speed, true_bearing=np.degrees(true) % 360, moving=moving,
                  **{f'fit_{k}': v[1] for k, v in fits.items()})
    return row, series


def classify_difference(r):
    """Rough label for the dominant behaviour of the DPVO-minus-true heading difference.

    A clear trend (>= 10 deg over the clip, R2 >= 0.5) is checked before the spread, so a small
    but steady drift is not hidden under 'constant'.
    """
    drift_ok = r['r2_drift'] >= 0.5 and abs(r['drift_over_clip_deg']) >= 10
    current_ok = r['r2_current'] >= 0.5 and r['current_mps'] <= 2.0
    if drift_ok and (not current_ok or r['r2_drift'] >= r['r2_current']):
        return 'drift'
    if current_ok:
        return 'current'
    if r['diff_std_deg'] < 5:
        return 'constant'
    if r['r2_both'] >= 0.5:
        return 'drift+current'
    return 'unexplained'


def plot_heading_difference(path, row, s, drift_pct=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1]))
    t = s['t']
    slow = ~s['moving']
    ax[0].plot(t, s['true_turned'], color='k', lw=1.2, label='true (direction of travel)')
    ax[0].plot(t, s['dpvo_turned'], color='tab:blue', lw=1.2, label='DPVO (camera forward)')
    ax[0].set_ylabel('turned since start (deg)\nclockwise +')
    ax[0].legend(loc='best', fontsize=8)
    ax[1].plot(t, s['diff'], color='tab:red', lw=1.2, label='DPVO - true')
    for name, style in (('fit_drift', ':'), ('fit_current', '--'), ('fit_both', '-.')):
        if name in s:
            ax[1].plot(t, s[name], color='0.3', ls=style, lw=1, label=name[4:] + ' fit')
    ax[1].axhline(0, color='0.7', lw=0.8)
    ax[1].set_ylabel('difference (deg)')
    ax[1].legend(loc='best', fontsize=8)
    ax[2].plot(t, s['speed'], color='tab:green', lw=1.2)
    ax[2].axhline(MIN_SPEED_MPS, color='0.7', lw=0.8, ls='--')
    ax[2].set_ylabel('true speed (m/s)')
    ax[2].set_xlabel('time since start (s)')
    for a in ax:  # shade where the boat is too slow for a reliable direction of travel
        a.fill_between(t, 0, 1, where=slow, transform=a.get_xaxis_transform(), color='0.85', lw=0)
    title = f"{row['key']} ({row['split']}, {row['camera_mode']})  label={row['label']}"
    if 'drift_deg_per_min' in row:
        title += (f"\ndrift {row['drift_deg_per_min']:+.1f} deg/min (R2 {row['r2_drift']:.2f}), "
                  f"current {row['current_mps']:.2f} m/s (R2 {row['r2_current']:.2f}), "
                  f"both R2 {row['r2_both']:.2f}")
    if drift_pct is not None:
        title += f"  |  DPVO head + const speed: {drift_pct:.1f}%"
    ax[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_heading_overview(path, rows, all_series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = sorted({r['camera_mode'] for r in rows})
    colors = dict(zip(modes, plt.cm.tab10.colors))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for r in rows:
        s = all_series[r['key']]
        d = np.where(s['moving'], s['diff'], np.nan)
        ax[0].plot(s['t'], d, color=colors[r['camera_mode']], lw=0.7, alpha=0.7)
    ax[0].set_xlabel('time since start (s)')
    ax[0].set_ylabel('DPVO - true heading (deg), moving samples')
    ax[0].set_title('heading difference, all clips')
    fitted = [r for r in rows if 'drift_deg_per_min' in r]
    for i, m in enumerate(modes):
        v = [r['drift_deg_per_min'] for r in fitted if r['camera_mode'] == m]
        ax[1].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v,
                      color=colors[m], s=18)
    ax[1].set_xticks(range(len(modes)), modes, rotation=30)
    ax[1].axhline(0, color='0.7', lw=0.8)
    ax[1].set_ylabel('linear drift (deg/min)')
    ax[1].set_title('yaw drift rate per clip')
    ax[2].scatter([r['r2_drift'] for r in fitted], [r['r2_current'] for r in fitted],
                  c=[colors[r['camera_mode']] for r in fitted], s=18)
    ax[2].plot([0, 1], [0, 1], color='0.7', lw=0.8)
    ax[2].set_xlabel('R2 drift model')
    ax[2].set_ylabel('R2 current model')
    ax[2].set_title('what explains the difference')
    for m in modes:
        ax[2].scatter([], [], color=colors[m], label=m)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heading_report(hd_rows):
    """Text summary of the heading-difference labels and drift rates."""
    from collections import Counter
    lines = ['== DPVO heading minus true heading (moving samples, true speed >= %.1f m/s)' % MIN_SPEED_MPS,
             'labels overall: ' + ', '.join(f'{k}={v}' for k, v in Counter(r['label'] for r in hd_rows).most_common())]
    for mode in sorted({r['camera_mode'] for r in hd_rows}):
        g = [r for r in hd_rows if r['camera_mode'] == mode]
        f = [r for r in g if 'drift_deg_per_min' in r]
        labels = ', '.join(f'{k}={v}' for k, v in Counter(r['label'] for r in g).most_common())
        stats = (f"median |drift| {np.median([abs(r['drift_deg_per_min']) for r in f]):.2f} deg/min, "
                 f"median diff std {np.median([r['diff_std_deg'] for r in f]):.1f} deg, "
                 f"median R2 drift/current/both {np.median([r['r2_drift'] for r in f]):.2f}/"
                 f"{np.median([r['r2_current'] for r in f]):.2f}/{np.median([r['r2_both'] for r in f]):.2f}"
                 if f else 'no fitted clips')
        lines.append(f'{mode:<10} n={len(g):<3} {labels} | {stats}')
    return '\n'.join(lines)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})


def true_speed_series(ref_t, ref_xyz, sigma_s=SMOOTH_SIGMA_S):
    """True speed (m/s) at reference-interval midpoints, smoothed like the headings."""
    _, tv = true_motion(ref_t, ref_xyz)
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    g, v = smooth_in_time(mid, tv, sigma_s)
    return g, v


def flow_diag(root, clip, speed_npz, out_png, med3_dirs=()):
    """Compare raw flow speeds (all / near / far bands) with the true speed for one clip, and
    score true heading + flow speed with a minimal filter (NaN -> clip median, 2 s rolling median)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from common import flow_speed_profile
    d = np.load(speed_npz)
    ref_t, ref_xyz = load_reference(root, clip)
    frame_t = load_timestamps(root, clip)
    gt, gv = true_speed_series(ref_t, ref_xyz)
    th, tv = true_motion(ref_t, ref_xyz)
    thf = at_frames_piecewise(ref_t, th, frame_t)
    row = dict(key=clip['key'], camera_mode=clip['camera_mode'])
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(gt, gv / np.median(gv), 'k', lw=2, label='true speed / median')
    for band, col in (('all', 'tab:blue'), ('near', 'tab:orange'), ('far', 'tab:green')):
        v = d[f'speed_{band}']
        ok = np.isfinite(v)
        tr = np.interp(d['t_mid'][ok], gt, gv)
        row[f'corr_{band}'] = float(np.corrcoef(v[ok], tr)[0, 1]) if ok.sum() > 10 else np.nan
        row[f'valid_{band}'] = float(ok.mean())
        ax[0].plot(d['t_mid'], v / np.nanmedian(v), color=col, lw=0.5, alpha=0.6,
                   label=f"flow {band} / median (r={row[f'corr_{band}']:.2f})")
    prof = flow_speed_profile(d, frame_t, band='all', window_s=2.0)
    row['drift_true_head_flow_speed'] = score(frame_t, integrate_planar(frame_t, thf, prof), ref_t, ref_xyz)
    row['drift_true_head_const'] = score(frame_t, integrate_planar(frame_t, thf, np.ones(len(frame_t))), ref_t, ref_xyz)
    row['drift_true_head_true_speed'] = score(frame_t, integrate_planar(frame_t, thf, at_frames_piecewise(ref_t, tv, frame_t)), ref_t, ref_xyz)
    ax[0].plot(frame_t, prof / np.median(prof), 'r', lw=1.2, label='flow profile (2 s median) / median')
    ax[0].set_ylim(0, 4)
    ax[0].set_ylabel('speed / clip median')
    ax[0].legend(fontsize=7, loc='upper left')
    rat = d['speed_near'] / d['speed_far']
    ax[1].plot(d['t_mid'], rat / np.nanmedian(rat), lw=0.5)
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel('near/far (norm.)')
    ax[1].set_xlabel('time (s)')
    ax[0].set_title(f"{clip['key']} ({clip['camera_mode']}): true head + flow speed "
                    f"{row['drift_true_head_flow_speed']:.1f}% vs const {row['drift_true_head_const']:.1f}% "
                    f"vs true speed {row['drift_true_head_true_speed']:.1f}%", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return row


# --------------------------------------------------------------------------------------
# Flow speed: tuning (train only) and evaluation
# --------------------------------------------------------------------------------------
TUNE_GRID = dict(agree_tol=[0.25, 0.4, 0.6, 1.0], window_s=[4.0, 8.0], clamp_lo=[0.0, 0.25], clamp_hi=[NO_LIMIT],
                 min_reliable_frac=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9], side_max=[NO_LIMIT, 0.3, 0.5, 0.7],
                 drop_backward=[False, True])
# Per-clip rules applied after the filter (cheap: per-clip features, no re-scoring).
GATE_CV = [0.0, 0.2, 0.25, 0.3, 0.35, 0.4]
GATE_HJUMP = [NO_LIMIT, 2.0, 2.5, 3.0, 3.5]


def speed_clip_data(root, clip, speed_npz, heading_pose_paths, sigma_s=SMOOTH_SIGMA_S):
    """Everything needed to score speed variants on one clip (references included)."""
    frame_t = load_timestamps(root, clip)
    ref_t, ref_xyz = load_reference(root, clip)
    th, tv = true_motion(ref_t, ref_xyz)
    runs = []
    for pp in heading_pose_paths:
        if pp.is_file():
            z = load_pose_npz(pp)
            runs.append((z['poses'], z['times']))
    med3, _ = multi_run_heading(runs, frame_t, sigma_s)
    d = dict(np.load(speed_npz)) if speed_npz is not None and speed_npz.is_file() else None
    return dict(key=clip['key'], split=clip['split'], camera_mode=clip['camera_mode'], frame_t=frame_t,
                ref_t=ref_t, ref_xyz=ref_xyz, thf=at_frames_piecewise(ref_t, th, frame_t),
                tvf=at_frames_piecewise(ref_t, tv, frame_t), med3=med3, d=d)


def _score_speed(data, heading, speed):
    if heading is None:
        return score(data['frame_t'], straight_line(data['frame_t']), data['ref_t'], data['ref_xyz'])
    return score(data['frame_t'], integrate_planar(data['frame_t'], heading, speed), data['ref_t'], data['ref_xyz'])


def _tune_clip(args):
    """All TUNE_GRID filter combinations on one clip. Returns (key, (flow scores [n_combos, 2]
    (true heading, med3 heading), constant-speed scores [2], flow CV [n_combos] (0 where the
    filter fell back to constant), horizon jump)). Per-clip gates are applied afterwards."""
    root, clip, speed_npz, pose_paths = args
    data = speed_clip_data(root, clip, speed_npz, pose_paths)
    ones = np.ones(len(data['frame_t']))
    const = np.array([_score_speed(data, data['thf'], ones), _score_speed(data, data['med3'], ones)])
    g = TUNE_GRID
    out, cvs = [], []
    for tol in g['agree_tol']:
        for side in g['side_max']:
            for back in g['drop_backward']:
                t, v, rel, n_rel = flow_reliable_speed(data['d'], tol, side, back)
                for win in g['window_s']:
                    sm = rolling_median(t, v, win) if n_rel >= 10 else None
                    for lo in g['clamp_lo']:
                        for hi in g['clamp_hi']:
                            sp = None
                            for mf in g['min_reliable_frac']:
                                if rel < mf or n_rel < 10:
                                    out.append(tuple(const))
                                    cvs.append(0.0)
                                    continue
                                if sp is None:
                                    sp, _ = flow_speed_from_reliable(t, v, data['frame_t'], win, lo, hi, smoothed=sm)
                                    sc = (_score_speed(data, data['thf'], sp), _score_speed(data, data['med3'], sp))
                                out.append(sc)
                                cvs.append(speed_variation(sp)[0])
    return clip['key'], (np.array(out), const, np.array(cvs), horizon_jump(data['d']))


def tune_speed(root, clips, speed_dir, pose_dirs, workers=8):
    """Fit filter_flow_speed thresholds and the per-clip gates on the given (TRAIN) clips:
    minimise the mean drift of med3 heading + flow speed. Returns (best params, rows, n clips)."""
    from concurrent.futures import ProcessPoolExecutor
    import itertools
    order = ['agree_tol', 'side_max', 'drop_backward', 'window_s', 'clamp_lo', 'clamp_hi', 'min_reliable_frac']
    combos = list(itertools.product(*(TUNE_GRID[k] for k in order)))   # same nesting as _tune_clip
    jobs = [(root, c, speed_dir / 'speed' / f"{c['key']}.npz", [d / 'poses' / f"{c['key']}.npz" for d in pose_dirs])
            for c in clips if (speed_dir / 'speed' / f"{c['key']}.npz").is_file()]
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_tune_clip, jobs))
    keys = sorted(res)
    S = np.stack([res[k][0] for k in keys])              # clips x combos x 2
    C = np.stack([res[k][1] for k in keys])              # clips x 2
    CV = np.stack([res[k][2] for k in keys])             # clips x combos
    HJ = np.array([res[k][3] for k in keys])             # clips
    rows = []
    for cvmin in GATE_CV:
        for hjmax in GATE_HJUMP:
            use = (CV >= cvmin) & (HJ[:, None] <= hjmax) & (CV > 0)
            G = np.where(use[..., None], S, C[:, None, :])
            mean = np.nanmean(G, axis=0)
            for i, combo in enumerate(combos):
                rows.append(dict(zip(order, combo), gate_stat='cv' if cvmin > 0 else 'none', gate_min=cvmin,
                                 horizon_jump_max=hjmax, mean_true_head_flow=float(mean[i, 0]),
                                 mean_med3_flow=float(mean[i, 1]), flow_used_frac=float(use[:, i].mean())))
    rows.sort(key=lambda r: r['mean_med3_flow'])
    best = {k: rows[0][k] for k in order + ['gate_stat', 'gate_min', 'horizon_jump_max']}
    return best, rows, len(keys)


def speed_eval_clip(args):
    root, clip, speed_npz, pose_paths, params = args
    data = speed_clip_data(root, clip, speed_npz, pose_paths)
    n = len(data['frame_t'])
    ones = np.ones(n)
    row = dict(key=clip['key'], split=clip['split'], camera_mode=clip['camera_mode'],
               true_head_const=_score_speed(data, data['thf'], ones),
               med3_const=_score_speed(data, data['med3'], ones),
               true_head_true_speed=_score_speed(data, data['thf'], data['tvf']),
               med3_true_speed=_score_speed(data, data['med3'], data['tvf']))
    if data['d'] is not None:
        sp, info = filter_flow_speed(data['d'], data['frame_t'], params)
        row.update(true_head_flow=_score_speed(data, data['thf'], sp),
                   med3_flow=_score_speed(data, data['med3'], sp), uses_flow=True, **info)
    else:  # no flow speed for this camera mode: constant speed
        row.update(true_head_flow=row['true_head_const'], med3_flow=row['med3_const'], uses_flow=False,
                   speed_fallback='not run', reliable_frac=np.nan, fallback_frac=np.nan)
    return row


SPEED_VARIANTS = ['true_head_const', 'true_head_flow', 'true_head_true_speed',
                  'med3_const', 'med3_flow', 'med3_true_speed']


DESIGN_CLIPS = ('clip_021', 'clip_038', 'clip_077')   # used to design the flow-speed method


def paired_report(rows, base='med3_const', new='med3_flow', tol=0.5):
    """Per-clip change of `new` vs `base` on the clips that use flow speed: how many get better /
    worse by more than tol points, by how much, and separately for clips where the baseline is
    already good (< 10 %). With and without the design clips, train and val separately."""
    lines = [f'== per clip: {new} vs {base} (clips using flow speed; change in drift points, '
             f'better/worse = beyond +-{tol})',
             f"{'group':<30}{'n':>4}{'better':>8}{'worse':>7}{'mean chg':>10}{'mean gain':>11}{'mean loss':>11}"
             f"{'worst loss':>12}"]
    for split in ('train', 'val', 'train+val'):
        for excl in (False, True):
            for good in (False, True):
                g = [r for r in rows if r['uses_flow'] and (split == 'train+val' or r['split'] == split)
                     and not (excl and r['key'] in DESIGN_CLIPS) and (not good or r[base] < 10)]
                if not g:
                    continue
                dlt = np.array([r[new] - r[base] for r in g])
                b, w = dlt < -tol, dlt > tol
                name = f"{split}{' -design' if excl else ''}{' base<10%' if good else ''}"
                lines.append(f'{name:<30}{len(g):>4}{b.sum():>8}{w.sum():>7}{dlt.mean():>+10.2f}'
                             f"{(dlt[b].mean() if b.any() else 0):>+11.2f}{(dlt[w].mean() if w.any() else 0):>+11.2f}"
                             f'{dlt.max():>+12.2f}')
    return '\n'.join(lines)


def speed_report(rows, worst_keys):
    def block(title, groups):
        lines = [f'== {title} (mean drift %)',
                 f"{'group':<18}{'n':>4}" + ''.join(f'{v:>15}' for v in SPEED_VARIANTS) + f"{'clip fallb.':>13}{'frame fallb.':>14}"]
        for name, g in groups:
            if not g:
                continue
            fl = [r for r in g if r['uses_flow']]
            clipfb = sum(r['speed_fallback'] == 'clip' for r in fl)
            fr = np.nanmean([1.0 if r['speed_fallback'] == 'clip' else r['fallback_frac'] for r in fl]) if fl else np.nan
            lines.append(f'{name:<18}{len(g):>4}' + ''.join(f"{np.nanmean([r[v] for r in g]):>15.2f}" for v in SPEED_VARIANTS)
                         + f'{clipfb:>8}/{len(fl):<4}' + (f'{fr:>14.3f}' if fl else f"{'-':>14}"))
        return '\n'.join(lines)
    modes = sorted({r['camera_mode'] for r in rows}, key=lambda m: -sum(r['camera_mode'] == m for r in rows))
    out = []
    for split in ('train', 'val'):
        g = [r for r in rows if r['split'] == split]
        out.append(block(f'{split}', [('all', g), ('all except zoom', [r for r in g if r['camera_mode'] != 'zoom']),
                                      ('all exc. design clips', [r for r in g if r['key'] not in DESIGN_CLIPS])]
                         + [(m, [r for r in g if r['camera_mode'] == m]) for m in modes]))
    out.append(block('train+val', [('all', rows), ('all except zoom', [r for r in rows if r['camera_mode'] != 'zoom'])]))
    out.append(paired_report(rows))
    w = [r for r in rows if r['key'] in worst_keys]
    out.append(block('15 worst clips of the earlier table', [('worst 15', w)]))
    lines = [f"{'clip':<10}{'split':<6}{'mode':<9}" + ''.join(f'{v:>15}' for v in SPEED_VARIANTS) + '  fallback']
    for r in sorted(w, key=lambda r: -r['med3_const']):
        lines.append(f"{r['key']:<10}{r['split']:<6}{r['camera_mode']:<9}" + ''.join(f'{r[v]:>15.1f}' for v in SPEED_VARIANTS)
                     + f"  {r['speed_fallback']} ({r['fallback_frac'] if r['uses_flow'] else float('nan'):.2f})")
    out.append('\n'.join(lines))
    return '\n\n'.join(out)


def flow_failure_diag(root, clip, speed_npz, params, out_png, frames_png=None, pose_dirs=()):
    """Why flow speed fails on a clip: flow (raw bands + filtered) vs true speed, band agreement,
    sideways fraction / fit residual, horizon correction; optionally frames with the water region
    at the moments of largest speed error. Returns the error times."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = dict(np.load(speed_npz))
    ref_t, ref_xyz = load_reference(root, clip)
    frame_t = load_timestamps(root, clip)
    gt, gv = true_speed_series(ref_t, ref_xyz)
    prof, info = filter_flow_speed(d, frame_t, params)
    raw, _ = filter_flow_speed(d, frame_t, dict(params, gate_stat='none', min_reliable_frac=0.0))
    tr = np.interp(frame_t, gt, gv)
    tr_n = tr / np.median(tr)
    t = d['t_mid']
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.3, 1]))
    ax[0].plot(gt, gv / np.median(gv), 'k', lw=2, label='true speed / median')
    for band, col in (('near', 'tab:orange'), ('far', 'tab:green')):
        v = d[f'speed_{band}']
        ax[0].plot(t, v / np.nanmedian(v), color=col, lw=0.4, alpha=0.5, label=f'flow {band} (raw)')
    ax[0].plot(frame_t, raw / np.median(raw), 'r', lw=1.3, label='flow, filtered (no gate)')
    ax[0].set_ylim(0, 4)
    ax[0].set_ylabel('speed / clip median')
    ax[0].legend(fontsize=7, loc='upper left', ncol=2)
    near, far = d['speed_near'], d['speed_far']
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log((near / np.nanmedian(near)) / (far / np.nanmedian(far)))
    ax[1].plot(t, lr, lw=0.4)
    ax[1].axhline(params['agree_tol'], color='r', lw=0.8)
    ax[1].axhline(-params['agree_tol'], color='r', lw=0.8)
    ax[1].set_ylim(-2, 2)
    ax[1].set_ylabel('log near/far')
    a, b = d['a_all'], d['b_all']
    side = np.abs(b) / np.hypot(a, b)
    ax[2].plot(t, rolling_median(t, side, 4.0), 'tab:purple', lw=1, label='sideways fraction (4 s median)')
    ax[2].plot(t, (a < 0).astype(float) * 0.95, 'k.', ms=1, label='backward motion')
    ax[2].set_ylim(0, 1)
    ax2 = ax[2].twinx()
    ax2.plot(t, rolling_median(t, d['rmse_all'], 4.0), 'tab:gray', lw=1)
    ax2.set_ylabel('fit rmse px', color='tab:gray')
    ax[2].legend(fontsize=7, loc='upper left')
    ax[3].plot(d['times'], d['horizon_pitch_deg'], label='horizon pitch corr.')
    ax[3].plot(d['times'], d['horizon_roll_deg'], label='roll corr.')
    ax[3].set_ylabel('deg')
    ax[3].legend(fontsize=7)
    ax[3].set_xlabel('time (s)')
    fq = flow_quality_features(d, params['agree_tol'])
    ax[0].set_title(f"{clip['key']} ({clip['split']}, {clip['camera_mode']}) decision={info['speed_fallback']}  "
                    f"CV {info.get('flow_cv', float('nan')):.2f}  reliable {fq['reliable_frac']:.2f}  "
                    f"side {fq['side_frac']:.2f}  backward {fq['backward_frac']:.2f}  rmse {fq['rmse_px']:.2f}px  "
                    f"horizon jump {fq['horizon_jump']:.1f}deg", fontsize=8)
    # moments of largest error (log ratio of normalised flow vs truth), >= 40 s apart
    err = np.abs(np.log(np.maximum(raw / np.median(raw), 1e-3) / np.maximum(tr_n, 1e-3)))
    picks = []
    for i in np.argsort(-err):
        if all(abs(frame_t[i] - frame_t[j]) > 40 for j in picks):
            picks.append(int(i))
        if len(picks) == 3:
            break
    for i in picks:
        for x in ax:
            x.axvline(frame_t[i], color='m', lw=0.8, ls='--')
    fig.tight_layout()
    fig.savefig(out_png, dpi=95)
    plt.close(fig)
    if frames_png is not None:
        render_flow_frames(root, clip, d, pose_dirs, [frame_t[i] for i in picks],
                           [(tr_n[i], raw[i] / np.median(raw)) for i in picks], frames_png)
    return [float(frame_t[i]) for i in picks]


def render_flow_frames(root, clip, d, pose_dirs, times_s, speeds, out_png):
    """Frames at times_s with horizon and water region, labelled with true and flow speed."""
    import flow_speed as fs
    fs.render_frames(root, clip, d, pose_dirs, times_s,
                     [f'true {tv:.2f} flow {fv:.2f} (x median)' for tv, fv in speeds], out_png)


def heading_seeds_plot(root, clip, pose_dirs, out_png, sigma_s=SMOOTH_SIGMA_S):
    """True heading vs each DPVO run's heading (turned since start, clockwise +) and their
    circular median, plus the true speed. Returns per-run stats."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame_t = load_timestamps(root, clip)
    ref_t, ref_xyz = load_reference(root, clip)
    th, tv = true_motion(ref_t, ref_xyz)
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    g, ths = smooth_in_time(mid, th, sigma_s)
    true_f = np.interp(frame_t, g, ths)
    start = frame_t <= frame_t[0] + START_WINDOW_S
    true_turn = np.degrees(true_f - true_f[start].mean())
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 1]))
    ax[0].plot(frame_t, true_turn, 'k', lw=2, label='true (direction of travel)')
    stats, runs = [], []
    for pd, col in zip(pose_dirs, ('tab:blue', 'tab:orange', 'tab:green')):
        pp = pd / 'poses' / f"{clip['key']}.npz"
        z = load_pose_npz(pp)
        runs.append((z['poses'], z['times']))
        poses, pt = clean_poses(z['poses'], z['times'])
        st = dict(run=pd.name, finite=int(np.isfinite(z['poses']).all(1).sum()), n=len(z['poses']))
        if poses is not None:
            h, _ = pose_heading(poses, pt)
            hf = np.degrees(heading_at_frames(pt, h, frame_t, sigma_s))
            ax[0].plot(frame_t, hf, color=col, lw=1, label=f'DPVO {pd.name}')
            dif = np.angle(np.exp(1j * np.radians(hf - true_turn)), deg=True)
            ax[1].plot(frame_t, dif, color=col, lw=1)
            st.update(fwd_cos=forward_motion_cosine(poses, pt), span_dpvo=float(np.ptp(hf)), span_true=float(np.ptp(true_turn)),
                      end_diff=float(dif[-50:].mean()))
        stats.append(st)
    hm, _ = multi_run_heading(runs, frame_t, sigma_s)
    if hm is not None:
        ax[0].plot(frame_t, np.degrees(hm), 'r--', lw=1.2, label='circular median')
    ax[0].set_ylabel('turned since start (deg), clockwise +')
    ax[0].legend(fontsize=7)
    ax[1].axhline(0, color='0.7')
    ax[1].set_ylabel('DPVO - true (deg)')
    gs, vs = true_speed_series(ref_t, ref_xyz)
    ax[2].plot(gs, vs, 'tab:green')
    ax[2].set_ylabel('true speed m/s')
    ax[2].set_xlabel('time (s)')
    ax[0].set_title(f"{clip['key']} ({clip['split']}, {clip['camera_mode']}): DPVO heading per seed vs true", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=95)
    plt.close(fig)
    return stats


def heading_scale(root, clip, pose_dirs, sigma_s=SMOOTH_SIGMA_S):
    """Least-squares slope of the median DPVO heading against the true heading (both turned since
    the start, smoothed alike). 1 = DPVO turns as much as the boat; 2 = twice as much (e.g. the
    focal length used is half the true one)."""
    frame_t = load_timestamps(root, clip)
    ref_t, ref_xyz = load_reference(root, clip)
    th, _ = true_motion(ref_t, ref_xyz)
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    g, ths = smooth_in_time(mid, th, sigma_s)
    true_f = np.interp(frame_t, g, ths)
    start = frame_t <= frame_t[0] + START_WINDOW_S
    tt = np.degrees(true_f - true_f[start].mean())
    runs = []
    for pd in pose_dirs:
        pp = pd / 'poses' / f"{clip['key']}.npz"
        if pp.is_file():
            z = load_pose_npz(pp)
            runs.append((z['poses'], z['times']))
    h, _ = multi_run_heading(runs, frame_t, sigma_s)
    if h is None or np.ptp(tt) < 15:
        return dict(key=clip['key'], camera_mode=clip['camera_mode'], split=clip['split'], scale=np.nan,
                    true_span=float(np.ptp(tt)))
    dh = np.degrees(h)
    slope = float((dh @ tt) / (tt @ tt))
    first = frame_t <= frame_t[0] + 120  # first 2 minutes (before late tracking failures)
    slope120 = float((dh[first] @ tt[first]) / (tt[first] @ tt[first])) if np.ptp(tt[first]) >= 15 else np.nan
    return dict(key=clip['key'], camera_mode=clip['camera_mode'], split=clip['split'], scale=slope,
                scale_first120=slope120, true_span=float(np.ptp(tt)))


def _focal_clip(args):
    root, clip, base_dirs, new_dirs, diag = args
    out = dict(key=clip['key'])
    for tag, dirs in (('f1', base_dirs), ('f2', new_dirs)):
        data = speed_clip_data(root, clip, None, [d / 'poses' / f"{clip['key']}.npz" for d in dirs])
        out[f'{tag}_med3_true_speed'] = _score_speed(data, data['med3'], data['tvf'])
        out[f'{tag}_med3_const'] = _score_speed(data, data['med3'], np.ones(len(data['frame_t'])))
        out['true_head_const'] = _score_speed(data, data['thf'], np.ones(len(data['frame_t'])))
    for scale, d in diag.items():
        pp = d / 'poses' / f"{clip['key']}.npz"
        if not pp.is_file():
            continue
        z = dict(np.load(pp))
        r = analyse_clip(root, clip, pp)
        out[f'diag{scale}_head_true_speed'] = r['dpvo_head_true_speed']
        for k in ('dpvo_residual_px', 'dpvo_weighted_residual_px', 'dpvo_mean_weight'):
            out[f'diag{scale}_{k}'] = float(z[k]) if k in z else np.nan
    return out


def focal_report(root, clips, base_dirs, new_dirs, diag, workers=8):
    """Step 1: 3-seed median at the new focal scale vs the base; step 2b: per diag scale, DPVO
    residual / confidence vs heading quality (true speed + DPVO heading) on the same runs."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_focal_clip, [(root, c, base_dirs, new_dirs, diag) for c in clips]))
    L = ['== step 1: 3-seed median heading, focal x1.0 vs new (drift %)',
         f"{'clip':<9}{'trueH+const':>12}{'x1 H+trueV':>11}{'new H+trueV':>12}{'x1 H+const':>11}{'new H+const':>12}"]
    for r in rows:
        L.append(f"{r['key']:<9}{r['true_head_const']:>12.1f}{r['f1_med3_true_speed']:>11.1f}{r['f2_med3_true_speed']:>12.1f}"
                 f"{r['f1_med3_const']:>11.1f}{r['f2_med3_const']:>12.1f}")
    for fn, name in ((np.mean, 'mean'), (np.median, 'median')):
        L.append(f"{name:<9}{fn([r['true_head_const'] for r in rows]):>12.2f}" + ''.join(
            f"{fn([r[k] for r in rows]):>{w}.2f}" for k, w in (('f1_med3_true_speed', 11), ('f2_med3_true_speed', 12),
                                                               ('f1_med3_const', 11), ('f2_med3_const', 12))))
    if diag:
        scales = list(diag)
        L += ['', '== step 2b: per focal scale (seed 0 diag runs): heading quality (DPVO head + true speed, %) / '
              'DPVO weighted residual (px) / mean confidence',
              f"{'clip':<9}" + ''.join(f'{"x" + s:>24}' for s in scales) + '   best-heading  lowest-residual  highest-conf']
        agree_res = agree_conf = 0
        for r in rows:
            cells = ''.join(f"{r.get(f'diag{s}_head_true_speed', np.nan):>8.1f}/{r.get(f'diag{s}_dpvo_weighted_residual_px', np.nan):>6.3f}/"
                            f"{r.get(f'diag{s}_dpvo_mean_weight', np.nan):>7.4f}" for s in scales)
            bh = min(scales, key=lambda s: r.get(f'diag{s}_head_true_speed', np.inf))
            br = min(scales, key=lambda s: r.get(f'diag{s}_dpvo_weighted_residual_px', np.inf))
            bc = max(scales, key=lambda s: r.get(f'diag{s}_dpvo_mean_weight', -np.inf))
            agree_res += bh == br
            agree_conf += bh == bc
            L.append(f"{r['key']:<9}{cells}   x{bh:<12}x{br:<16}x{bc}")
        L.append(f'lowest residual picks the best-heading scale on {agree_res}/{len(rows)} clips; '
                 f'highest confidence on {agree_conf}/{len(rows)}')
    return '\n'.join(L), rows


ZOOM_SCALES = ('1.0', '1.5', '2.0', '2.5')
ZOOM_REPEAT = {'1.0': ['runs/dpvo-fps8-seed0', 'runs/dpvo-fps8-seed1', 'runs/dpvo-fps8-seed2'],
               '1.5': ['runs/dpvo-fps8-seed0-f1.5'], '2.0': ['runs/dpvo-fps8-seed0-f2', 'runs/dpvo-fps8-seed1-f2',
                                                         'runs/dpvo-fps8-seed2-f2'], '2.5': ['runs/dpvo-fps8-seed0-f2.5']}


def _zoom_clip(args):
    """Per clip: residual/confidence/heading per diag scale, repeat-run heading quality per scale
    (same setting, other runs), turning, rocking, scene and texture features, DPVO failures."""
    import cv2
    import flow_speed as fs
    root, clip = args
    k = clip['key']
    out = dict(key=k)
    for sc in ZOOM_SCALES:
        pp = REPO_ROOT / f'runs/diag-f{sc}/poses/{k}.npz'
        z = dict(np.load(pp))
        out[f'res_{sc}'] = float(z['dpvo_weighted_residual_px'])
        out[f'conf_{sc}'] = float(z['dpvo_mean_weight'])
        out[f'head_{sc}'] = analyse_clip(root, clip, pp)['dpvo_head_true_speed']
        out[f'finite_{sc}'] = float(np.isfinite(z['poses']).all(1).mean())
        reps = [analyse_clip(root, clip, REPO_ROOT / d / 'poses' / f'{k}.npz')['dpvo_head_true_speed'] for d in ZOOM_REPEAT[sc]]
        out[f'head_rep_{sc}'] = reps                     # other runs of the same scale (repeatability)
    # true turning
    ref_t, ref_xyz = load_reference(root, clip)
    th, _ = true_motion(ref_t, ref_xyz)
    mid = 0.5 * (ref_t[1:] + ref_t[:-1])
    g, ths = smooth_in_time(mid, th, SMOOTH_SIGMA_S)
    turn = float(np.degrees(np.abs(np.diff(ths)).sum()))
    out.update(turn_total_deg=turn, turn_per_min=turn / ((g[-1] - g[0]) / 60), true_span_deg=float(np.degrees(np.ptp(ths))))
    # rocking: pitch/roll of the camera about DPVO's up at x2.0 (closest to the true focal), 5 s high-pass
    z = load_pose_npz(REPO_ROOT / f'runs/dpvo-fps8-seed0-f2/poses/{k}.npz')
    ok = np.isfinite(z['poses']).all(1)
    R = quat_to_rotmat(z['poses'][ok, 3:7])
    U = estimate_up(R)
    up_c = np.einsum('nji,j->ni', R, U)                   # up in camera coords
    pitch = np.degrees(np.arctan2(-up_c[:, 2], -up_c[:, 1]))
    roll = np.degrees(np.arctan2(up_c[:, 0], -up_c[:, 1]))
    t = z['times'][ok]
    out['rock_pitch_std_deg'] = float(np.std(pitch - rolling_median(t, pitch, 5.0)))
    out['rock_roll_std_deg'] = float(np.std(roll - rolling_median(t, roll, 5.0)))
    # scene: water area below the fitted horizon (DPVO picks patch centres uniformly at random,
    # so this is also the expected fraction of patches on water) and texture near the image edges
    d = dict(np.load(REPO_ROOT / f'runs/flow-v2/speed/{k}.npz'))
    frames, dec, times, K = fs.clip_frames(root, clip, 8.0)
    zr = load_pose_npz(REPO_ROOT / f'runs/dpvo-fps8-seed0/poses/{k}.npz')
    Rr = quat_to_rotmat(np.where(np.isfinite(zr['poses'][:, 3:7]), zr['poses'][:, 3:7], [0, 0, 0, 1]))
    Ur = estimate_up(Rr[np.isfinite(zr['poses']).all(1)])
    H, W = frames[0].shape
    xs = np.arange(W, dtype=float)
    yy = np.arange(H)[:, None]
    wf, edge_out, edge_all, tex_above = [], [], [], []
    for i in np.linspace(0, len(frames) - 1, 24).astype(int):
        u = fs.correction(d['horizon_pitch_deg'][i], d['horizon_roll_deg'][i]) @ (Rr[i].T @ Ur)
        hy = fs.horizon_rows(u / np.linalg.norm(u), K, xs)
        below = yy > np.nan_to_num(hy, nan=0)[None, :]
        wf.append(below.mean())
        gm = np.hypot(cv2.Sobel(frames[i], cv2.CV_32F, 1, 0), cv2.Sobel(frames[i], cv2.CV_32F, 0, 1))
        edge_all.append(gm.mean())
        edge_out.append(np.concatenate([gm[:, :W // 7], gm[:, -W // 7:]], 1).mean())
        tex_above.append(gm[~below].sum() / max(gm.sum(), 1e-9))
    out.update(water_frac=float(np.mean(wf)), edge_texture_ratio=float(np.mean(edge_out) / np.mean(edge_all)),
               texture_share_above_horizon=float(np.mean(tex_above)), mean_gradient=float(np.mean(edge_all)))
    return out


def zoom_analysis(root, clips, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_zoom_clip, [(root, c) for c in clips]))


def path_compare(root, clip, variants, out_png):
    """True path vs median-heading paths of several DPVO run sets (label -> run dirs), each
    integrated with constant speed (left) and with the true speed (right) and Sim(3)-aligned to the
    reference exactly as the evaluator does. Returns {label: (drift const, drift true speed)}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    out = {}
    ref_drawn = False
    for (label, dirs), col in zip(variants.items(), ('tab:blue', 'tab:red', 'tab:green')):
        data = speed_clip_data(root, clip, None, [d / 'poses' / f"{clip['key']}.npz" for d in dirs])
        res = []
        for a, speed, name in ((ax[0], np.ones(len(data['frame_t'])), 'constant speed'), (ax[1], data['tvf'], 'true speed')):
            xyz = integrate_planar(data['frame_t'], data['med3'], speed)
            vo = metrics.score_trajectory(data['frame_t'], xyz, data['ref_t'], data['ref_xyz'])
            if not ref_drawn:
                a.plot(*vo['reference'].T, 'k', lw=2.2, label='true path')
                a.plot(*vo['reference'][0], 'ko')
            a.plot(*vo['aligned'].T, color=col, lw=1.3, label=f"{label} heading: drift {vo['drift_pct']:.1f} %")
            a.set_title(f"{clip['key']} ({clip['split']}, {clip['camera_mode']}): median heading + {name}, Sim(3)-aligned", fontsize=9)
            res.append(float(vo['drift_pct']))
        ref_drawn = True
        out[label] = tuple(res)
    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel('east (m)')
        a.set_ylabel('north (m)')
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out


def _compare_clip(args):
    root, clip, runs, params = args
    row = dict(key=clip['key'], split=clip['split'], camera_mode=clip['camera_mode'])
    for label, (pose_dirs, speed_dir) in runs.items():
        pps = [Path(d) / 'poses' / f"{clip['key']}.npz" for d in pose_dirs]
        pps = [pp for pp in pps if pp.is_file()]
        sp = Path(speed_dir) / 'speed' / f"{clip['key']}.npz" if speed_dir else None
        if not pps:
            row[f'{label}_head'] = row[f'{label}_pipe'] = np.nan
            continue
        data = speed_clip_data(root, clip, sp, pps)
        row[f'{label}_head'] = _score_speed(data, data['med3'], data['tvf'])
        speed = np.ones(len(data['frame_t']))
        if data['d'] is not None:
            fs_, info = filter_flow_speed(data['d'], data['frame_t'], params)
            speed = fs_ if info['speed_fallback'] == 'none' else speed
        row[f'{label}_pipe'] = _score_speed(data, data['med3'], speed)
        row[f'{label}_nan'] = any(not np.isfinite(load_pose_npz(pp)['poses']).all() for pp in pps)
    return row


def compare_runs_heading(root, clips, runs, params, workers=8):
    """Single DPVO runs side by side (label -> (pose dir, flow speed dir or None)): heading only
    (run heading + true speed) and pipeline (run heading + filtered flow speed, else constant)."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_compare_clip, [(root, c, runs, params) for c in clips]))
    labels = list(runs)
    base = labels[0]
    L = []
    for metric, name in (('head', 'heading only (run heading + true speed)'), ('pipe', 'pipeline (run heading + flow v3a)')):
        L.append(f'== {name}, drift %')
        L.append(f"{'group':<14}{'n':>4}" + ''.join(f'{l + " mean/med":>24}' for l in labels) + '   better/worse vs ' + base)
        groups = [('all', rows)] + [(sp, [r for r in rows if r['split'] == sp]) for sp in ('train', 'val')]
        groups += [(m, [r for r in rows if r['camera_mode'] == m]) for m in sorted({r['camera_mode'] for r in rows})]
        for gname, g in groups:
            if not g:
                continue
            cells = ''.join(f"{np.nanmean([r[f'{l}_{metric}'] for r in g]):>12.2f}/{np.nanmedian([r[f'{l}_{metric}'] for r in g]):<11.2f}" for l in labels)
            bw = '  '.join(f"{l}: {sum(r[f'{l}_{metric}'] < r[f'{base}_{metric}'] - .5 for r in g)}/{sum(r[f'{l}_{metric}'] > r[f'{base}_{metric}'] + .5 for r in g)}"
                           for l in labels[1:])
            L.append(f'{gname:<14}{len(g):>4}' + cells + '   ' + bw)
    L.append('')
    L.append(f"{'clip':<9}{'split':<6}" + ''.join(f'{l + " head/pipe":>22}' for l in labels))
    for r in rows:
        L.append(f"{r['key']:<9}{r['split']:<6}" + ''.join(
            f"{r[f'{l}_head']:>10.1f}/{r[f'{l}_pipe']:<6.1f}{'*' if r.get(f'{l}_nan') else ' ':<5}" for l in labels))
    L.append('(* = run with NaN poses)')
    return '\n'.join(L), rows


def _combo_clip(args):
    """One wide clip: DPVO seeds, their median, VGGT, and the 4-member median; GT-free disagreement
    features; heading-only and pipeline (flow v3a) scores per heading source."""
    from common import _run_heading, circular_median
    root, clip, dpvo_dirs, vggt_dir, speed_dir, params = args
    k = clip['key']
    data = speed_clip_data(root, clip, speed_dir / 'speed' / f'{k}.npz', [d / 'poses' / f'{k}.npz' for d in dpvo_dirs])
    t = data['frame_t']
    seeds = []
    for d in dpvo_dirs:
        z = load_pose_npz(d / 'poses' / f'{k}.npz')
        h, _ = _run_heading(z['poses'], z['times'], t, SMOOTH_SIGMA_S)
        if h is not None:
            seeds.append(h)
    zv = load_pose_npz(vggt_dir / 'poses' / f'{k}.npz')
    hv, _ = _run_heading(zv['poses'], zv['times'], t, SMOOTH_SIGMA_S)
    med3 = data['med3']
    speed = np.ones(len(t))
    if data['d'] is not None:
        fs_, info = filter_flow_speed(data['d'], t, params)
        speed = fs_ if info['speed_fallback'] == 'none' else speed
    row = dict(key=k, split=clip['split'])
    heads = dict(med3=med3, vggt=hv, med4=circular_median(seeds + [hv]) if hv is not None else med3)
    for name, h in heads.items():
        row[f'{name}_head'] = _score_speed(data, h, data['tvf'])
        row[f'{name}_pipe'] = _score_speed(data, h, speed)
    wrap = lambda x: np.degrees(np.angle(np.exp(1j * x)))
    row['seed_mad_deg'] = float(np.median([np.median(np.abs(wrap(h - med3))) for h in seeds])) if seeds else np.nan
    row['seed_max_deg'] = float(max(np.percentile(np.abs(wrap(h - med3)), 95) for h in seeds)) if seeds else np.nan
    if hv is not None:
        dd = wrap(med3 - hv)
        dd = dd - np.median(dd)
        row['dv_p95_deg'] = float(np.percentile(np.abs(dd), 95))
        row['dv_win60_deg'] = heading_divergence_60s(t, med3, hv)
    return row


COMBO_FEATURES = ('seed_mad_deg', 'seed_max_deg', 'dv_p95_deg', 'dv_win60_deg')


def combo_eval(root, clips, dpvo_dirs, vggt_dir, speed_dir, params, workers=8):
    """VGGT/DPVO heading combinations on wide train+val: fixed variants and a per-clip switch
    (VGGT if a GT-free feature exceeds a threshold) fitted on TRAIN."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_combo_clip, [(root, c, dpvo_dirs, vggt_dir, speed_dir, params) for c in clips]))
    tr = [r for r in rows if r['split'] == 'train']
    fits = []
    for feat in COMBO_FEATURES:
        vals = sorted({round(r[feat], 2) for r in tr if np.isfinite(r[feat])})
        for thr in vals + [np.inf]:
            for metric in ('head', 'pipe'):
                pick = lambda r: r[f'vggt_{metric}'] if r[feat] > thr else r[f'med3_{metric}']
                fits.append(dict(feature=feat, threshold=thr, metric=metric, train_mean=float(np.mean([pick(r) for r in tr]))))
    best = {m: min([f for f in fits if f['metric'] == m], key=lambda f: f['train_mean']) for m in ('head', 'pipe')}
    for r in rows:
        for m, f in best.items():
            r[f'switch_{m}'] = r[f'vggt_{m}'] if r[f['feature']] > f['threshold'] else r[f'med3_{m}']
            r[f'switch_{m}_uses_vggt'] = bool(r[f['feature']] > f['threshold'])
    L = []
    for m, name in (('head', 'heading only (heading + true speed)'), ('pipe', 'full pipeline (heading + flow v3a)')):
        f = best[m]
        L.append(f"== {name}; switch fitted on train: VGGT if {f['feature']} > {f['threshold']} "
                 f"(train mean {f['train_mean']:.2f})")
        L.append(f"{'group':<8}{'n':>4}" + ''.join(f'{v + " mean/med":>20}' for v in ('med3', 'vggt', 'med4', 'switch'))
                 + f"{'switch uses VGGT':>18}")
        for gname, g in (('train', tr), ('val', [r for r in rows if r['split'] == 'val']), ('all', rows)):
            if not g:
                continue
            cells = ''.join(f"{np.mean([r[f'{v}_{m}'] for r in g]):>11.2f}/{np.median([r[f'{v}_{m}'] for r in g]):<8.2f}"
                            for v in ('med3', 'vggt', 'med4', 'switch'))
            L.append(f'{gname:<8}{len(g):>4}' + cells + f"{sum(r[f'switch_{m}_uses_vggt'] for r in g):>12}/{len(g)}")
        for v in ('vggt', 'med4', 'switch'):
            for gname, g in (('train', tr), ('val', [r for r in rows if r['split'] == 'val'])):
                if not g:
                    continue
                d = np.array([r[f'{v}_{m}'] - r[f'med3_{m}'] for r in g])
                L.append(f'   {v:<7}{gname:<6} vs med3: better {int((d < -.5).sum())}, worse {int((d > .5).sum())}, '
                         f'mean change {d.mean():+.2f}, worst loss {d.max():+.1f}')
    L.append('\nworst clips (heading only, med3 DPVO):')
    L.append(f"{'clip':<9}{'split':<6}" + ''.join(f'{v:>8}' for v in ('med3', 'vggt', 'med4', 'switch')) + ''.join(f'{c[:12]:>13}' for c in COMBO_FEATURES))
    for r in sorted(rows, key=lambda r: -r['med3_head'])[:15]:
        L.append(f"{r['key']:<9}{r['split']:<6}" + ''.join(f"{r[f'{v}_head']:>8.1f}" for v in ('med3', 'vggt', 'med4', 'switch'))
                 + ''.join(f"{r[c]:>13.1f}" for c in COMBO_FEATURES))
    return '\n'.join(L), rows


def _motion_clip(args):
    """Heading from camera orientation vs from direction of motion (DPVO 3-seed and VGGT), each with
    the true speed (heading-only drift), plus turn ratios against the true heading."""
    from common import circular_median, _run_heading
    root, clip, dpvo_dirs, vggt_dir = args
    k = clip['key']
    data = speed_clip_data(root, clip, None, [d / 'poses' / f'{k}.npz' for d in dpvo_dirs])
    t = data['frame_t']
    row = dict(key=k, split=clip['split'], camera_mode=clip['camera_mode'])
    true_turn = np.unwrap(data['thf'])

    def stats(h, name):
        row[f'{name}_head'] = _score_speed(data, h, data['tvf'])
        tt = np.degrees(true_turn - true_turn[0]); hh = np.degrees(h - h[0])
        row[f'{name}_ratio'] = float((hh @ tt) / (tt @ tt)) if np.ptp(tt) >= 15 else np.nan

    def motion_at_frames(poses, pt):
        poses, pt = clean_poses(poses, pt)
        if poses is None:
            return None
        mh = motion_heading(poses, pt, SMOOTH_SIGMA_S)
        if mh is None:
            return None
        g, h, _ = mh
        return np.interp(t, g, h)

    stats(data['med3'], 'dpvo_orient')
    mot = [motion_at_frames(z['poses'], z['times']) for z in (load_pose_npz(d / 'poses' / f'{k}.npz') for d in dpvo_dirs)]
    mot = [m for m in mot if m is not None]
    if mot:
        stats(circular_median(mot) if len(mot) > 1 else mot[0], 'dpvo_motion')
    zv = load_pose_npz(vggt_dir / 'poses' / f'{k}.npz')
    hv, _ = _run_heading(zv['poses'], zv['times'], t, SMOOTH_SIGMA_S)
    if hv is not None:
        stats(hv, 'vggt_orient')
    mv = motion_at_frames(zv['poses'], zv['times'])
    if mv is not None:
        stats(mv, 'vggt_motion')
    return row


def _failure_job(args):
    return flow_failure_diag(*args)


def group_means(rows, key_fn):
    out = {}
    for row in rows:
        out.setdefault(key_fn(row), []).append(row)
    table = {}
    for name, group in sorted(out.items()):
        entry = dict(n=len(group))
        for v in VARIANTS:
            vals = np.array([r[v] for r in group], float)
            vals = vals[np.isfinite(vals)]
            entry[v] = dict(mean=float(vals.mean()) if len(vals) else None,
                            median=float(np.median(vals)) if len(vals) else None, n=int(len(vals)))
        table[name] = entry
    return table


def format_table(title, table):
    short = {'straight': 'straight', 'raw_dpvo': 'raw', 'dpvo_head_const': 'dH+cV',
             'dpvo_head_inv_const': 'dHinv+cV', 'dpvo_motion_const': 'dMot+cV',
             'dpvo_head_true_speed': 'dH+tV', 'true_head_const': 'tH+cV',
             'true_head_dpvo_speed': 'tH+dV', 'true_head_true_speed': 'tH+tV'}
    lines = [f'== {title} (mean drift %, [n] when fewer clips scored)',
             f"{'group':<12}{'n':>4}" + ''.join(f'{short[v]:>11}' for v in VARIANTS)]
    for name, e in table.items():
        cells = ''
        for v in VARIANTS:
            m = e[v]['mean']
            cell = '-' if m is None else f'{m:.2f}' + ('' if e[v]['n'] == e['n'] else f"[{e[v]['n']}]")
            cells += f'{cell:>11}'
        lines.append(f'{name:<12}{e["n"]:>4}' + cells)
    return '\n'.join(lines)


def focal_lower_bound(clips, runs):
    """Oracle focal lower bound: per clip, the focal run with the lowest official drift.

    `runs` maps a label (focal scale) to a predictions run scored by starter/evaluate.py
    (its results.json); the first label is the x1.0 base. A clip missing from a run (no sweep
    for its camera type) keeps the base. Returns (text, rows)."""
    res = {lab: {r['key']: r['drift_pct'] for r in read_json(d / 'results.json') if r.get('drift_pct') is not None}
           for lab, d in runs.items()}
    base = next(iter(runs))
    rows = []
    for c in clips:
        k = c['key']
        cand = {lab: r[k] for lab, r in res.items() if k in r}
        best = min(cand, key=cand.get)
        rows.append(dict(key=k, split=c['split'], camera_mode=c['camera_mode'], base=cand[base], best_label=best,
                         best=cand[best], **{f'drift_{lab}': cand.get(lab, np.nan) for lab in runs}))
    L = [f"== oracle focal lower bound (per clip best of {', '.join(runs)}; base {base})",
         f"{'group':<10}{'n':>4}{'base mean/med':>18}{'best mean/med':>18}  best focal counts"]
    groups = [('all', rows)] + [(m, [r for r in rows if r['camera_mode'] == m])
                                for m in sorted({r['camera_mode'] for r in rows})]
    for name, g in groups:
        b, o = [r['base'] for r in g], [r['best'] for r in g]
        counts = {lab: sum(r['best_label'] == lab for r in g) for lab in runs}
        L.append(f"{name:<10}{len(g):>4}{np.mean(b):>11.2f}/{np.median(b):<6.2f}{np.mean(o):>11.2f}/{np.median(o):<6.2f}"
                 f"  {', '.join(f'{lab}:{n}' for lab, n in counts.items() if n)}")
    L.append(f"with perfect focal, dev drift would be {np.mean([r['best'] for r in rows]):.2f} % "
             f"instead of {np.mean([r['base'] for r in rows]):.2f} % (mean over {len(rows)} clips)")
    L.append('')
    L.append(f"{'clip':<9}{'mode':<9}" + ''.join(f'{lab:>8}' for lab in runs) + '    best')
    for r in rows:
        if sum(np.isfinite(r[f'drift_{lab}']) for lab in runs) > 1:
            L.append(f"{r['key']:<9}{r['camera_mode']:<9}" + ''.join(f"{r[f'drift_{lab}']:>8.2f}" for lab in runs)
                     + f"    {r['best_label']}")
    return '\n'.join(L), rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val', help='train and/or val (test has no references)')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--poses', type=Path, help='run_dpvo.py output directory')
    ap.add_argument('--predictions', type=Path, help='optional predictions run dir to score with evaluate_clip')
    ap.add_argument('--output', required=True, type=Path, help='report directory')
    ap.add_argument('--sigma', type=float, default=SMOOTH_SIGMA_S)
    ap.add_argument('--heading-diff', action='store_true',
                    help='also analyse/plot DPVO heading minus true heading over time (needs --poses)')
    ap.add_argument('--flow-diag', type=Path, help='flow_speed.py output dir: plot/score flow speed vs truth')
    ap.add_argument('--speed-tune', type=Path, help='flow_speed.py output dir: fit the speed filter on TRAIN clips')
    ap.add_argument('--speed-eval', type=Path, help='flow_speed.py output dir: evaluate speed variants (with --speed-params)')
    ap.add_argument('--speed-params', type=Path, help='speed filter thresholds JSON (from --speed-tune)')
    ap.add_argument('--heading-poses', type=Path, nargs='+', help='run_dpvo.py dirs for the median heading (e.g. 3 seeds)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--motion-check', type=Path, help='VGGT run dir: orientation vs motion-direction heading')
    ap.add_argument('--combo-vggt', type=Path, help='VGGT run dir: evaluate VGGT/DPVO heading combinations (wide)')
    ap.add_argument('--compare-runs', nargs='+',
                    help='LABEL=POSEDIR[+POSEDIR...][:SPEEDDIR]; several pose dirs = circular-median heading (first = base)')
    ap.add_argument('--zoom-analysis', action='store_true', help='flir-VIS focal: why the residual detector works or not')
    ap.add_argument('--focal-base', type=Path, nargs='+', help='focal report: base run dirs (e.g. 3 seeds at x1.0)')
    ap.add_argument('--focal-new', type=Path, nargs='+', help='focal report: run dirs at the new focal scale')
    ap.add_argument('--focal-diag', nargs='*', default=[], help='focal report: SCALE=DIR diag runs (run_dpvo --dpvo-diag)')
    ap.add_argument('--focal-lower-bound', nargs='+',
                    help='LABEL=PREDDIR (scored by evaluate.py), first = x1.0 base: per-clip best focal (oracle)')
    ap.add_argument('--worst', type=int, default=0,
                    help='also list the N worst clips, classify speed/heading/DPVO failure (needs --heading-diff)')
    args = ap.parse_args(argv)
    if args.worst and not args.heading_diff:
        ap.error('--worst needs --heading-diff (it uses the heading series for DPVO failures)')
    if args.heading_diff and not args.poses:
        ap.error('--heading-diff needs --poses')
    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    if any(c['reference'] is None for c in selected):
        ap.error('oracle needs references; test clips have none')
    root = args.dataset.resolve().parent
    if args.motion_check:
        from concurrent.futures import ProcessPoolExecutor
        dirs = [REPO_ROOT / f'runs/dpvo-fps8-seed{s}' for s in range(3)]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(_motion_clip, [(root, c, dirs, args.motion_check) for c in selected]))
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'motion_check.csv', rows)
        V = ('dpvo_orient', 'dpvo_motion', 'vggt_orient', 'vggt_motion')
        print(f"{'clip':<9}" + ''.join(f'{v + " drift/ratio":>24}' for v in V))
        for r in rows:
            print(f"{r['key']:<9}" + ''.join(f"{r.get(v + '_head', np.nan):>15.1f} / {r.get(v + '_ratio', np.nan):>+5.2f}" for v in V))
        print(f"{'mean':<9}" + ''.join(f"{np.nanmean([r.get(v + '_head', np.nan) for r in rows]):>15.2f}{'':>9}" for v in V))
        print(f"{'median':<9}" + ''.join(f"{np.nanmedian([r.get(v + '_head', np.nan) for r in rows]):>15.2f}{'':>9}" for v in V))
        return 0
    if args.combo_vggt:
        params = read_json(args.speed_params) if args.speed_params else None
        text, rows = combo_eval(root, selected, [REPO_ROOT / f'runs/dpvo-fps8-seed{s}' for s in range(3)],
                                args.combo_vggt, REPO_ROOT / 'runs/flow-v2', params, args.workers)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'combo.csv', rows)
        (args.output / 'combo.txt').write_text(text + '\n')
        print(text)
        return 0
    if args.compare_runs:
        runs = {}
        for spec in args.compare_runs:
            label, rest = spec.split('=', 1)
            pose, _, speed = rest.partition(':')
            runs[label] = ([REPO_ROOT / d for d in pose.split('+')], (REPO_ROOT / speed) if speed else None)
        params = read_json(args.speed_params) if args.speed_params else None
        text, rows = compare_runs_heading(root, selected, runs, params, args.workers)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'compare_runs.csv', rows)
        (args.output / 'compare_runs.txt').write_text(text + '\n')
        print(text)
        return 0
    if args.focal_lower_bound:
        runs = {kv.split('=', 1)[0]: Path(kv.split('=', 1)[1]) for kv in args.focal_lower_bound}
        text, rows = focal_lower_bound(selected, runs)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'focal_lower_bound.csv', rows)
        (args.output / 'focal_lower_bound.txt').write_text(text + '\n')
        print(text)
        return 0
    if args.zoom_analysis:
        rows = zoom_analysis(root, selected, args.workers)
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'zoom_analysis.json', rows)
        print('written', args.output / 'zoom_analysis.json')
        return 0
    if args.focal_base:
        diag = {kv.split('=', 1)[0]: Path(kv.split('=', 1)[1]) for kv in args.focal_diag}
        text, rows = focal_report(root, selected, args.focal_base, args.focal_new, diag, args.workers)
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'focal_report.csv', rows)
        (args.output / 'focal_report.txt').write_text(text + '\n')
        print(text)
        return 0
    if args.speed_tune:
        train = [c for c in selected if c['split'] == 'train']
        if len(train) != len(selected):
            ap.error('--speed-tune fits on TRAIN clips only; select --split train')
        best, table, n = tune_speed(root, train, args.speed_tune, args.heading_poses, args.workers)
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'speed_params.json', best)
        write_csv(args.output / 'speed_tune.csv', table)
        print(f'tuned on {n} train clips; best (mean med3+flow {table[0]["mean_med3_flow"]:.2f}%, '
              f'true head+flow {table[0]["mean_true_head_flow"]:.2f}%): {best}')
        for r in table[:8]:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        return 0
    if args.speed_eval:
        from concurrent.futures import ProcessPoolExecutor
        params = read_json(args.speed_params) if args.speed_params else None
        jobs = [(root, c, args.speed_eval / 'speed' / f"{c['key']}.npz",
                 [d / 'poses' / f"{c['key']}.npz" for d in args.heading_poses], params) for c in selected]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(speed_eval_clip, jobs))
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / 'speed_eval.csv', rows)
        worst = set()
        prev = REPO_ROOT / 'runs' / 'oracle' / 'per_clip.csv'  # the earlier seed-0 oracle table
        if prev.is_file():
            old = list(csv.DictReader(open(prev)))
            worst = {r['key'] for r in sorted(old, key=lambda r: -float(r['dpvo_head_const']))[:15]}
        report = speed_report(rows, worst)
        (args.output / 'speed_eval.txt').write_text(report + '\n')
        print(report)
        return 0
    if args.flow_diag and args.speed_params:  # failure analysis with the tuned filter
        from concurrent.futures import ProcessPoolExecutor
        params = read_json(args.speed_params)
        args.output.mkdir(parents=True, exist_ok=True)
        jobs = [(root, c, args.flow_diag / 'speed' / f"{c['key']}.npz", params, args.output / f"diag_{c['key']}.png",
                 args.output / f"frames_{c['key']}.jpg", args.heading_poses or ()) for c in selected]
        with ProcessPoolExecutor(min(args.workers, len(jobs))) as ex:
            for c, picks in zip(selected, ex.map(_failure_job, jobs)):
                print(c['key'], 'largest-error times', [round(x) for x in picks], flush=True)
        return 0
    if args.flow_diag:
        args.output.mkdir(parents=True, exist_ok=True)
        diag = []
        for clip in selected:
            sp = args.flow_diag / 'speed' / f"{clip['key']}.npz"
            if sp.is_file():
                r = flow_diag(root, clip, sp, args.output / f"flow_{clip['key']}.png")
                diag.append(r)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        write_csv(args.output / 'flow_diag.csv', diag)
        return 0
    rows, hd_rows, hd_series = [], [], {}
    hd_dir = args.output / 'heading_diff'
    if args.heading_diff:
        hd_dir.mkdir(parents=True, exist_ok=True)
    for clip in selected:
        pose_path = args.poses / 'poses' / f"{clip['key']}.npz" if args.poses else None
        row = analyse_clip(root, clip, pose_path, args.sigma)
        if args.heading_diff:
            hd = heading_difference(root, clip, pose_path, args.sigma)
            if hd is not None:
                hrow, series = hd
                hrow['dpvo_head_const'] = row['dpvo_head_const']
                hrow['dpvo_head_true_speed'] = row['dpvo_head_true_speed']
                hd_rows.append(hrow)
                hd_series[clip['key']] = series
                np.savez(hd_dir / f"{clip['key']}.npz", **series)
                plot_heading_difference(hd_dir / f"{clip['key']}.png", hrow, series, row['dpvo_head_const'])
        if args.predictions:
            rec, _ = evaluate_clip(root, clip, args.predictions / 'predictions' / f"{clip['key']}.json")
            row.update(submitted_status=rec['status'], submitted_drift=rec.get('drift_pct'),
                       submitted_overlap_cov=rec.get('temporal_coverage'),
                       submitted_max_gap_s=rec.get('max_prediction_gap_s'))
        rows.append(row)
        print(f"{row['key']} {row['split']:<5} {row['camera_mode']:<9} "
              + ' '.join(f"{v}={row[v]:.2f}" for v in ('straight', 'raw_dpvo', 'dpvo_head_const',
                                                       'true_head_true_speed')), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_clip.csv', rows)
    summary = dict(overall=group_means(rows, lambda r: 'all'),
                   split=group_means(rows, lambda r: r['split']),
                   camera_mode=group_means(rows, lambda r: r['camera_mode']))
    write_json(args.output / 'summary.json', summary)
    report = '\n\n'.join(format_table(t, summary[k]) for t, k in
                         (('overall', 'overall'), ('per split', 'split'), ('per camera_mode', 'camera_mode')))
    if hd_rows:
        write_csv(args.output / 'heading_diff.csv', hd_rows)
        plot_heading_overview(args.output / 'heading_diff_overview.png', hd_rows, hd_series)
        report += '\n\n' + heading_report(hd_rows)
    if args.worst:
        report += '\n\n' + worst_report(rows, hd_series, args.worst)
        write_csv(args.output / 'per_clip.csv', rows)  # now with category columns
    (args.output / 'summary.txt').write_text(report + '\n')
    print('\n' + report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
