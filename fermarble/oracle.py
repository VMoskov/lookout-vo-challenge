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

from common import (DEFAULT_DATASET, SMOOTH_SIGMA_S, clean_poses, gap_aware_coverage,
                    forward_motion_cosine, heading_at_frames, heading_constant_speed, integrate_planar, load_manifest,
                    load_pose_npz, load_timestamps, motion_heading, pose_heading, select_clips,
                    straight_line, write_json)
import metrics  # starter
from evaluate import evaluate_clip  # starter
from public_dataset import load_reference  # starter

VARIANTS = ['straight', 'raw_dpvo', 'dpvo_head_const', 'dpvo_head_inv_const', 'dpvo_motion_const',
            'dpvo_head_true_speed', 'true_head_const', 'true_head_dpvo_speed', 'true_head_true_speed']
STATIONARY_MPS = 0.05


def true_motion(ref_t, ref_xyz):
    """Per-reference-interval heading (unwrapped, rad) and speed (m/s) of the planar reference."""
    vel = np.diff(ref_xyz[:, :2], axis=0) / np.diff(ref_t)[:, None]
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
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
        row['raw_dpvo'] = s(raw['poses'][:, :3], raw['times'])
        row['raw_overlap_cov'] = float(metrics.overlap_window(raw['times'], ref_t).mean())
        poses, pt = clean_poses(raw['poses'], raw['times'])
        if poses is not None:
            row['fwd_cos'] = forward_motion_cosine(poses, pt, sigma_s)
            row['fwd_cos_inv'] = forward_motion_cosine(poses, pt, sigma_s, invert=True)
            heading, _ = pose_heading(poses)
            if heading is not None:
                dh = heading_at_frames(pt, heading, frame_t, sigma_s)
                row['dpvo_head_true_speed'] = s(integrate_planar(frame_t, dh, tv))
            mh = motion_heading(poses, pt, sigma_s)
            if mh is not None:
                grid, mhead, mspeed = mh
                row['dpvo_motion_const'] = s(integrate_planar(frame_t, np.interp(frame_t, grid, mhead), ones))
                row['true_head_dpvo_speed'] = s(integrate_planar(frame_t, th, np.interp(frame_t, grid, mspeed)))
    return row


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
    args = ap.parse_args(argv)
    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    if any(c['reference'] is None for c in selected):
        ap.error('oracle needs references; test clips have none')
    root = args.dataset.resolve().parent
    rows = []
    for clip in selected:
        pose_path = args.poses / 'poses' / f"{clip['key']}.npz" if args.poses else None
        row = analyse_clip(root, clip, pose_path, args.sigma)
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
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.output / 'per_clip.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    summary = dict(overall=group_means(rows, lambda r: 'all'),
                   split=group_means(rows, lambda r: r['split']),
                   camera_mode=group_means(rows, lambda r: r['camera_mode']))
    write_json(args.output / 'summary.json', summary)
    report = '\n\n'.join(format_table(t, summary[k]) for t, k in
                         (('overall', 'overall'), ('per split', 'split'), ('per camera_mode', 'camera_mode')))
    (args.output / 'summary.txt').write_text(report + '\n')
    print('\n' + report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
