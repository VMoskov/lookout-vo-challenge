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

from common import (DEFAULT_DATASET, SMOOTH_SIGMA_S, START_WINDOW_S, clean_poses, gap_aware_coverage,
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
