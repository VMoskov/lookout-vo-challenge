#!/usr/bin/env python3
"""Relative boat speed from optical flow on the water surface (CPU; never reads references).

For each clip, on the same ~8 fps frames DPVO tracked (starter ClipPreprocessor, same stride):

1. Dense optical flow between consecutive frames (OpenCV DIS, MEDIUM preset) on gray frames.
2. Water region (pluggable, see `water_region`): pixels at least MIN_DEPRESSION_DEG below the
   horizon, minus pixels that never move (own hull, poles, overlays: the automatic ego mask).
   The horizon comes from DPVO's per-frame attitude plus one per-clip mount correction
   (pitch, roll) fitted to the strongest horizontal image edge along the predicted horizon
   (`fit_horizon_correction`), so it follows rocking and roll.
3+4. Water-plane model with the rotation removed exactly: a water pixel with ray r (K^-1 p,
   r_z = 1) and up vector u (camera coords) lies at X = r / s * h with s = -u.r; between
   frames i and j the camera moves by c (horizontal, in units of the unknown camera height h)
   and rotates by DPVO's R_ji = R_j^T R_i, so the pixel goes to proj(R_ji (r/s - c)). The 2
   unknowns of c are fitted to the observed flow by robust Gauss-Newton (Huber IRLS).
   Speed = |c| / dt, relative within a clip (h unknown; Sim(3) absorbs the scale).
5. Separately for a near band (lower half of the water pixels by depression angle) and a
   far band (upper half), plus all water pixels.

Writes <output>/speed/<clip>.npz with per-pair raw measurements; filtering into a speed
profile is common.filter_flow_speed (thresholds fitted on train only, by oracle.py).
"""
from __future__ import annotations
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import (DEFAULT_DATASET, asset_path, estimate_up, load_manifest, load_pose_npz,
                    load_timestamps, quat_to_rotmat, select_clips, write_json)

GRID_STEP = 8                 # px between flow samples on the (half-res) DPVO frame
MIN_DEPRESSION_DEG = 1.5      # water starts this far below the horizon
MAX_DEPRESSION_DEG = 70.0
STATIC_FLOW_PX = 0.3          # 80th-percentile |flow| below this = never moves = ego/overlay
HORIZON_SAMPLES = 48          # frames used for the coarse per-clip horizon correction
HORIZON_WINDOW_S = 20.0       # then refitted locally in windows of this length
MIN_BAND_PIXELS = 40
HUBER_PX = 1.0


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def correction(pitch_deg, roll_deg):
    """Mount correction applied to the camera-frame up vector (pitch about x, roll about z)."""
    return rot_z(np.radians(roll_deg)) @ rot_x(np.radians(pitch_deg))


def horizon_rows(u, K, xs):
    """Image row of the horizon (rays with u.r = 0) at columns xs, or NaN where undefined."""
    fx, fy, cx, cy = K
    with np.errstate(divide='ignore', invalid='ignore'):
        y = cy - fy * (u[0] * (xs - cx) / fx + u[2]) / u[1]
    return np.where(np.abs(u[1]) > 1e-3, y, np.nan)


def edge_map(frame):
    """|I(y-3) - I(y+3)| of a lightly blurred frame: large across the horizon."""
    b = cv2.GaussianBlur(frame, (0, 0), 1.5).astype(np.float32)
    d = np.zeros_like(b)
    d[3:-3] = np.abs(b[:-6] - b[6:])
    return d


def search_horizon(edges, ups, K, pitches, rolls, base=(0.0, 0.0)):
    """Best (pitch, roll) correction (degrees, added to `base`) putting the predicted horizon on
    the strongest edge in the given frames. Returns (pitch, roll, score)."""
    H, W = edges[0].shape
    xs = np.arange(4, W - 4, 8, dtype=float)
    xi = xs.astype(int)
    best = (base[0], base[1], -1.0)
    for dp in pitches:
        for dr in rolls:
            C = correction(base[0] + dp, base[1] + dr)
            tot, n = 0.0, 0
            for d, u in zip(edges, ups):
                y = horizon_rows(C @ u, K, xs)
                ok = np.isfinite(y) & (y >= 4) & (y < H - 4)
                if ok.sum() < 0.5 * len(xs):
                    continue
                tot += d[y[ok].astype(int), xi[ok]].mean()
                n += 1
            if n >= 0.5 * len(edges) and tot / n > best[2]:
                best = (float(base[0] + dp), float(base[1] + dr), tot / n)
    return best


def fit_horizon_track(frames, ups, times, K, ok):
    """Per-frame mount correction (pitch, roll) of DPVO's up vector: a coarse per-clip fit, then
    a local refit every HORIZON_WINDOW_S seconds (DPVO's pitch/roll drift slowly), smoothed by a
    3-window median and interpolated to every frame. Returns (pitch[n], roll[n], window table)."""
    n = len(frames)
    sample = np.linspace(0, n - 1, HORIZON_SAMPLES).astype(int)
    sample = sample[ok[sample]]
    edges = {i: edge_map(frames[i]) for i in sample}
    gp, gr, _ = search_horizon([edges[i] for i in sample], ups[sample], K,
                               np.arange(-20, 20.01, 0.5), np.arange(-8, 8.01, 0.5))
    centres, wp, wr = [], [], []
    for t0 in np.arange(times[0], times[-1], HORIZON_WINDOW_S):
        idx = np.nonzero((times >= t0) & (times < t0 + HORIZON_WINDOW_S) & ok)[0]
        if len(idx) < 6:
            continue
        idx = idx[np.linspace(0, len(idx) - 1, 6).astype(int)]
        p, r, _ = search_horizon([edges[i] if i in edges else edge_map(frames[i]) for i in idx],
                                 ups[idx], K, np.arange(-4, 4.01, 0.25), np.arange(-2, 2.01, 0.5), base=(gp, gr))
        centres.append(times[idx].mean())
        wp.append(p)
        wr.append(r)
    if not centres:
        return np.full(n, gp), np.full(n, gr), np.array([[times.mean(), gp, gr]])
    med3 = lambda v: np.array([np.median(v[max(0, k - 1):k + 2]) for k in range(len(v))])
    wp, wr = med3(np.array(wp)), med3(np.array(wr))
    return (np.interp(times, centres, wp), np.interp(times, centres, wr),
            np.column_stack([centres, wp, wr]))


def water_region(rays, u, static, min_dep_deg=MIN_DEPRESSION_DEG, max_dep_deg=MAX_DEPRESSION_DEG):
    """Pluggable water mask at the sample points: below the horizon by [min, max] degrees and not
    static. Replace or AND with a semantic water mask (e.g. PanSR) here. Returns (mask, s)."""
    s = -(rays @ u)
    dep = np.degrees(np.arcsin(np.clip(s / np.linalg.norm(rays, axis=1), -1, 1)))
    return (dep >= min_dep_deg) & (dep <= max_dep_deg) & ~static, s


def project(K, X):
    fx, fy, cx, cy = K
    return np.column_stack([fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy])


def fit_translation(p, flow, rays, s, R_ji, e1, e2, K, iters=6):
    """Robust Gauss-Newton for the horizontal camera displacement c = a e1 + b e2 (units of
    camera height) that best explains `flow` at pixels p under the water-plane model.
    Returns (a, b, rmse_px, n_inliers)."""
    X = rays / s[:, None]                      # water points, camera height = 1
    ab = np.zeros(2)
    target = p + flow
    w = np.ones(len(p))

    def predict(v):
        Xj = (X - (v[0] * e1 + v[1] * e2)) @ R_ji.T
        return project(K, Xj), Xj[:, 2]

    for _ in range(iters):
        pred, z = predict(ab)
        r = (target - pred).ravel()
        J = np.empty((len(r), 2))
        for k in range(2):  # numeric Jacobian (2 unknowns)
            dv = np.zeros(2)
            dv[k] = 1e-4
            J[:, k] = ((predict(ab + dv)[0] - pred).ravel()) / 1e-4
        res = np.abs(r).reshape(-1, 2).max(1)
        w = np.where(res <= HUBER_PX, 1.0, HUBER_PX / np.maximum(res, 1e-9))
        ww = np.repeat(w, 2)
        A = J.T @ (J * ww[:, None])
        if np.linalg.cond(A) > 1e12:
            return np.nan, np.nan, np.nan, 0
        ab = ab + np.linalg.solve(A, J.T @ (ww * r))
    pred, z = predict(ab)
    err = np.linalg.norm(target - pred, axis=1)
    ok = (z > 0) & (err < 5 * HUBER_PX)
    return float(ab[0]), float(ab[1]), float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else np.nan, int(ok.sum())


# --------------------------------------------------------------------------------------
# Per clip
# --------------------------------------------------------------------------------------
def clip_frames(root, clip, target_fps, focal_scale=1.0):
    """Gray frames, decoded indices, times and intrinsics exactly as DPVO saw them (with the same
    focal_scale on the released fx, fy as run_dpvo.py --focal-scale)."""
    from preprocessing import ClipPreprocessor  # starter
    frame_times = load_timestamps(root, clip)
    calibration = dict(clip['calibration'])
    if focal_scale != 1.0:
        calibration.update(fx=calibration['fx'] * focal_scale, fy=calibration['fy'] * focal_scale)
    pre = ClipPreprocessor(str(asset_path(root, clip['video'])), clip['camera_mode'], target_fps,
                           calibration=calibration, frame_timestamps=frame_times)
    frames, dec, times, K = [], [], [], None
    for _, img, intr, pts, d in pre.frames_with_time():
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        dec.append(d)
        times.append(pts)
        K = tuple(float(v) for v in intr)
    return frames, np.array(dec), np.array(times), K


def best_pose_run(pose_paths):
    """Of several DPVO runs (seeds) of a clip, the one with the most finite poses."""
    runs = [(int(np.isfinite(z['poses']).all(1).sum()), str(pp), z)
            for pp in pose_paths if pp.is_file() for z in [load_pose_npz(pp)]]
    if not runs:
        raise FileNotFoundError('no pose file for this clip')
    n, path, z = max(runs, key=lambda r: r[0])
    return z, path


def process_clip(root, clip, pose_paths, out_path, target_fps=8.0, focal_scale=1.0):
    """Flow + water-plane speed for one clip. Returns a summary dict."""
    t0 = time.time()
    frames, dec, times, K = clip_frames(root, clip, target_fps, focal_scale)
    t_dec = time.time() - t0
    z, pose_used = best_pose_run(pose_paths)
    if not np.array_equal(z['frame_indices'], dec):
        raise ValueError('DPVO frames differ from the decoded frames')
    R = quat_to_rotmat(np.nan_to_num(z['poses'][:, 3:7], nan=0.0) + np.where(
        np.isfinite(z['poses'][:, 3:7]).all(1, keepdims=True), 0, [0, 0, 0, 1]))
    pose_ok = np.isfinite(z['poses']).all(1)
    U = estimate_up(R[pose_ok])                     # world up (DPVO frame), as for the heading
    ups = np.einsum('nji,j->ni', R, U)              # camera-frame up per frame: R^T U

    H, W = frames[0].shape
    gy, gx = np.mgrid[GRID_STEP // 2:H:GRID_STEP, GRID_STEP // 2:W:GRID_STEP]
    p = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    fx, fy, cx, cy = K
    rays = np.column_stack([(p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, np.ones(len(p))])

    # 1. flow for every consecutive pair, sampled on the grid
    t1 = time.time()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    n = len(frames)
    flows = np.empty((n - 1, len(p), 2), np.float32)
    for k in range(n - 1):
        F = dis.calc(frames[k], frames[k + 1], None)
        flows[k] = F[gy, gx].reshape(-1, 2)
    t_flow = time.time() - t1

    # 2. horizon correction (per frame) and ego/static mask
    hp, hr, htable = fit_horizon_track(frames, ups, times, K, pose_ok)
    static = np.percentile(np.linalg.norm(flows, axis=2), 80, axis=0) < STATIC_FLOW_PX
    del frames

    # 3-5. per pair: water pixels, near/far bands, robust fit
    keys = ('a_all', 'b_all', 'rmse_all', 'n_all', 'a_near', 'b_near', 'rmse_near', 'n_near',
            'a_far', 'b_far', 'rmse_far', 'n_far')
    out = {k: np.full(n - 1, np.nan) for k in keys}
    for k in range(n - 1):
        if not (pose_ok[k] and pose_ok[k + 1]):
            continue
        u = correction(hp[k], hr[k]) @ ups[k]
        u /= np.linalg.norm(u)
        water, s = water_region(rays, u, static)
        if water.sum() < 2 * MIN_BAND_PIXELS:
            continue
        R_ji = R[k + 1].T @ R[k]
        fwd = np.array([0, 0, 1.0]) - u[2] * u      # camera forward, flattened
        e1 = fwd / np.linalg.norm(fwd)
        e2 = np.cross(e1, u)                        # right (camera +x side)
        split = np.median(s[water])
        for band, sel in (('all', water), ('near', water & (s >= split)), ('far', water & (s < split))):
            if sel.sum() < MIN_BAND_PIXELS:
                continue
            a, b, rmse, ninl = fit_translation(p[sel], flows[k][sel].astype(float), rays[sel], s[sel],
                                               R_ji, e1, e2, K)
            out[f'a_{band}'][k], out[f'b_{band}'][k] = a, b
            out[f'rmse_{band}'][k], out[f'n_{band}'][k] = rmse, ninl
    dt = np.diff(times)
    speeds = {f'speed_{b}': np.hypot(out[f'a_{b}'], out[f'b_{b}']) / dt for b in ('all', 'near', 'far')}
    t_mid = 0.5 * (times[:-1] + times[1:])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, t_mid=t_mid, dt=dt, times=times, frame_indices=dec, **out, **speeds,
             horizon_pitch_deg=hp, horizon_roll_deg=hr, horizon_windows=htable,
             static_frac=float(static.mean()), K=np.array(K), focal_scale=np.float64(focal_scale))
    return dict(key=clip['key'], camera_mode=clip['camera_mode'], n_pairs=n - 1, poses=pose_used,
                focal_scale=focal_scale,
                valid_frac=float(np.isfinite(speeds['speed_all']).mean()),
                horizon_pitch_deg=[round(float(np.min(hp)), 2), round(float(np.max(hp)), 2)],
                horizon_roll_deg=[round(float(np.min(hr)), 2), round(float(np.max(hr)), 2)],
                static_frac=float(static.mean()),
                decode_s=round(t_dec, 1), flow_s=round(t_flow, 1), total_s=round(time.time() - t0, 1))


def render_frames(root, clip, d, pose_dirs, times_s, labels, out_png, width=640):
    """Frames at times_s with the fitted horizon (red) and the water sample points (green).
    Never reads references (labels are supplied by the caller)."""
    frames, dec, times, K = clip_frames(root, clip, 8.0, float(d.get('focal_scale', 1.0)))
    z, _ = best_pose_run([pd / 'poses' / f"{clip['key']}.npz" for pd in pose_dirs])
    R = quat_to_rotmat(np.where(np.isfinite(z['poses'][:, 3:7]), z['poses'][:, 3:7], [0, 0, 0, 1]))
    U = estimate_up(R[np.isfinite(z['poses']).all(1)])
    H, W = frames[0].shape
    gy, gx = np.mgrid[GRID_STEP // 2:H:GRID_STEP, GRID_STEP // 2:W:GRID_STEP]
    p = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    fx, fy, cx, cy = K
    rays = np.column_stack([(p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, np.ones(len(p))])
    tiles = []
    for ts, label in zip(times_s, labels):
        k = int(np.argmin(np.abs(times - ts)))
        u = correction(d['horizon_pitch_deg'][k], d['horizon_roll_deg'][k]) @ (R[k].T @ U)
        u /= np.linalg.norm(u)
        water, _ = water_region(rays, u, np.zeros(len(p), bool))
        im = cv2.cvtColor(frames[k], cv2.COLOR_GRAY2BGR)
        for (x, y) in p[water].astype(int):
            cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
        xs = np.arange(0, W, 3.0)
        for x, y in zip(xs, horizon_rows(u, K, xs)):
            if np.isfinite(y) and 0 <= y < H:
                cv2.circle(im, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.putText(im, f"{clip['key']} {clip['camera_mode']} t={times[k]:.0f}s {label}", (8, 28), 0, 0.8, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, (width, int(width * H / W))))
    w = max(t.shape[1] for t in tiles)
    cv2.imwrite(str(out_png), np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]))


def _render_job(args):
    root, clip, npz, pose_dirs, out_png = args
    d = dict(np.load(npz))
    ts = np.quantile(d['times'], [0.15, 0.5, 0.85])
    render_frames(root, clip, d, pose_dirs, ts, ['' for _ in ts], out_png, width=480)
    return clip['key']


def _job(args):
    root, clip, pose_paths, out_path, focal_scale = args
    try:
        return process_clip(root, clip, pose_paths, out_path, focal_scale=focal_scale)
    except Exception as exc:
        return dict(key=clip['key'], camera_mode=clip['camera_mode'], error=str(exc)[:300])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--modes', default='wide', help='camera modes to process (comma-separated, or all)')
    ap.add_argument('--poses', required=True, type=Path, nargs='+',
                    help='run_dpvo.py output dir(s); per clip the run with most finite poses gives the rotations')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--focal-scale', type=float, default=1.0,
                    help='multiply the released fx, fy (must match the run_dpvo.py --focal-scale of --poses)')
    ap.add_argument('--render', action='store_true',
                    help='only render a geometry check (3 frames with horizon/water) per processed clip')
    args = ap.parse_args(argv)
    cv2.setNumThreads(2)
    root = args.dataset.resolve().parent
    clips = select_clips(load_manifest(args.dataset), args.split, args.clips)
    if args.modes != 'all':
        clips = [c for c in clips if c['camera_mode'] in args.modes.split(',')]
    if args.render:
        out_dir = args.output / 'geometry'
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(root, c, args.output / 'speed' / f"{c['key']}.npz", args.poses, out_dir / f"{c['key']}.jpg")
                for c in clips if (args.output / 'speed' / f"{c['key']}.npz").is_file()]
        with ProcessPoolExecutor(args.workers) as ex:
            for k in ex.map(_render_job, jobs):
                print('rendered', k, flush=True)
        return 0
    jobs = []
    for c in clips:
        out = args.output / 'speed' / f"{c['key']}.npz"
        if out.is_file() and not args.overwrite:
            continue
        jobs.append((root, c, [d / 'poses' / f"{c['key']}.npz" for d in args.poses], out, args.focal_scale))
    print(f'{len(jobs)} clips to process ({len(clips) - len(jobs)} already done)', flush=True)
    summary_path = args.output / 'flow_speed_summary.json'
    summary = {}
    if summary_path.is_file():
        import json
        summary = json.loads(summary_path.read_text())
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(_job, jobs):
            summary[res['key']] = res
            write_json(summary_path, summary)
            print(res, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
