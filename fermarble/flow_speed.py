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

from common import (DEFAULT_DATASET, asset_path, estimate_up, load_manifest, load_pose_npz, load_timestamps,
                    quat_to_rotmat, select_clips, smooth_in_time, write_json)

GRID_STEP = 8                 # px between flow samples on the (half-res) DPVO frame
MIN_DEPRESSION_DEG = 1.5      # water starts this far below the horizon
MAX_DEPRESSION_DEG = 70.0
STATIC_FLOW_PX = 0.3          # 80th-percentile |flow| below this = never moves = ego/overlay
HORIZON_SAMPLES = 48          # frames used for the coarse per-clip horizon correction
HORIZON_WINDOW_S = 20.0       # then refitted locally in windows of this length
MIN_BAND_PIXELS = 40
STILL_REL = 0.25              # --drop-still: |flow| below this x the median |flow| of its grid row in that pair ...
STILL_RUN = 16                # ... for at least this many consecutive pairs (2 s at 8 fps) = lens droplet / static spot
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


def resampled_rotations(path, times):
    """Camera-to-world rotations of another heading source (e.g. Pi3X keyframes) slerp-interpolated to
    `times`; frames outside its pose span are marked not ok. Returns (R [n,3,3], ok [n])."""
    from scipy.spatial.transform import Rotation, Slerp
    z = load_pose_npz(path)
    good = np.isfinite(z['poses']).all(1)
    t, q = z['times'][good], z['poses'][good, 3:7]
    ok = (times >= t[0]) & (times <= t[-1])
    R = np.tile(np.eye(3), (len(times), 1, 1))
    R[ok] = Slerp(t, Rotation.from_quat(q))(times[ok]).as_matrix()
    return R, ok


def still_runs(flows, gy_rows, rel=STILL_REL, run=STILL_RUN):
    """(pairs, points) mask of samples whose flow stays far below that of their image row (|flow| <
    rel x the row median) for >= run consecutive pairs: lens droplets and other spots that do not move
    with the scene, which the clip-wide static mask misses when they come and go."""
    mag = np.linalg.norm(flows, axis=2)                                    # pairs x points
    rows = np.unique(gy_rows)
    rowmed = np.empty_like(mag)
    for r in rows:
        sel = gy_rows == r
        rowmed[:, sel] = np.median(mag[:, sel], axis=1, keepdims=True)
    still = mag < np.maximum(rel * rowmed, 0.02)
    c = np.concatenate([np.zeros((1, still.shape[1])), np.cumsum(still, 0)])
    full = (c[run:] - c[:-run]) == run                                     # window [k, k+run) all still
    f = np.concatenate([np.zeros((1, still.shape[1])), np.cumsum(full, 0)])
    out = np.zeros_like(still)
    for k in range(still.shape[0]):                                        # pair k inside any full window
        lo, hi = max(0, k - run + 1), min(full.shape[0], k + 1)
        out[k] = (f[hi] - f[lo]) > 0 if hi > lo else False
    return out


SEG_WATER = 10                # LaRS class id of water in seg_run.py masks
SEG_SKY = 11
SEG_MIN_COLS = 0.3            # segmentation horizon: fit a line only if >= this fraction of columns has water
SEG_SIGMA_S = 1.0             # ... then median over 9 segmented frames and Gaussian smoothing in time
HS_CAND_PX = 5.0              # horizon sanity (hsU variants): windows where DPVO's horizon is > this below the water edge ...
HS_CAND_CROSS = 0.1           # ... or crosses non-water below the edge in > this fraction of columns are refitted


def seg_water_points(seg, dec, p):
    """(frames, points) bool: PanSR water at the grid points, from the nearest segmented frame."""
    if not np.array_equal(seg['frame_indices'], dec):
        raise ValueError('segmentation frames differ from the decoded frames')
    q = int(seg['quarter'])
    m = seg['masks']
    yi = np.clip((p[:, 1] // q).astype(int), 0, m.shape[1] - 1)
    xi = np.clip((p[:, 0] // q).astype(int), 0, m.shape[2] - 1)
    sf = seg['seg_frames']
    nearest = np.clip(np.searchsorted(sf, np.arange(len(dec))), 0, len(sf) - 1)
    lower = np.clip(nearest - 1, 0, len(sf) - 1)
    nearest = np.where(np.abs(sf[lower] - np.arange(len(dec))) < np.abs(sf[nearest] - np.arange(len(dec))), lower, nearest)
    return m[:, yi, xi][nearest] == SEG_WATER


def seg_horizon_lines(seg, H, sky_only=False):
    """Per segmented frame, a robust line (slope, intercept) through the top edge of the water region
    (water/sky or water/shore boundary): least squares, then 3 rounds dropping residuals beyond
    max(3 MAD, 2 px). NaN where too few columns have water or the fit keeps < half of them.
    `sky_only`: use only columns where the pixel just above the water edge is sky (the water/sky
    boundary = the true horizon; skips columns where water meets shore or obstacles)."""
    W = seg['water_top'].shape[1]
    xs = np.arange(W, dtype=float)
    out = np.full((len(seg['water_top']), 2), np.nan)
    q = int(seg['quarter'])
    qcols = np.minimum(np.arange(W) // q, seg['masks'].shape[2] - 1)
    for i, top in enumerate(seg['water_top']):
        ok = (top > 0) & (top < H - 1)
        if sky_only:
            above = seg['masks'][i, np.clip(top // q - 1, 0, None), qcols]
            ok &= above == SEG_SKY
        if ok.mean() < SEG_MIN_COLS:
            continue
        x, y = xs[ok], top[ok].astype(float)
        keep = np.ones(len(x), bool)
        for _ in range(3):
            m, c = np.polyfit(x[keep], y[keep], 1)
            r = np.abs(y - (m * x + c))
            keep = r <= max(3 * 1.4826 * np.median(r[keep]), 2.0)
        if keep.mean() >= 0.5:
            out[i] = np.polyfit(x[keep], y[keep], 1)
    return out


def seg_horizon_ups(seg, times, K, W, H, sky_only=False, shift_px=0.0):
    """Camera-frame up vector per frame whose horizon (rays with u.r = 0) is the segmentation line,
    median-filtered over 9 segmented frames and smoothed in time (SEG_SIGMA_S). None if no frame
    has a line. The line is the visible water edge: the true horizon at open sea, the shoreline
    (below the horizon) in harbours. `shift_px` moves the line up by that many pixels."""
    lines = seg_horizon_lines(seg, H, sky_only)
    lines[:, 1] -= shift_px
    ok = np.isfinite(lines).all(1)
    if ok.sum() < 3:
        return None, lines
    ts = times[seg['seg_frames']][ok]
    L = lines[ok]
    med = np.array([np.median(L[max(0, i - 4):i + 5], axis=0) for i in range(len(L))])
    g, sm = smooth_in_time(ts, med, SEG_SIGMA_S) if len(ts) > 1 else (ts, med)
    slope = np.interp(times, g, sm[:, 0])
    icpt = np.interp(times, g, sm[:, 1])
    fx, fy, cx, cy = K
    ray = lambda x, y: np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], -1)
    r1, r2 = ray(np.zeros_like(slope), icpt), ray(np.full_like(slope, W - 1.0), slope * (W - 1) + icpt)
    u = np.cross(r1, r2)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u[u[:, 1] > 0] *= -1                                    # up = camera -y side
    return u, lines


def horizon_sanity_stats(seg, dpvo_up, K, times, H, W):
    """Per HORIZON_WINDOW_S window (from the first frame): median over the segmented frames of how far (px) DPVO's
    (mount-corrected) horizon lies below the PanSR water edge (median over columns), and of the fraction of columns
    where it crosses non-water, non-sky pixels below that edge (own boat, obstacles). Returns (window starts,
    below, cross, window index per frame)."""
    q = int(seg['quarter'])
    m = seg['masks']
    xs = np.arange(4, W - 4, 8, dtype=float)
    xi = xs.astype(int)
    t0s = np.arange(times[0], times[-1], HORIZON_WINDOW_S)
    win = np.clip(((times - times[0]) // HORIZON_WINDOW_S).astype(int), 0, len(t0s) - 1)
    below, cross = [[] for _ in t0s], [[] for _ in t0s]
    for j, f in enumerate(seg['seg_frames']):
        y = horizon_rows(dpvo_up[f], K, xs)
        top = seg['water_top'][j][xi].astype(float)
        ok = np.isfinite(y) & (top > 0) & (top < H - 1)
        if ok.sum() < SEG_MIN_COLS * len(xs):
            continue
        yy = np.clip(y[ok], 0, H - 1).astype(int)
        cls = m[j, np.clip(yy // q, 0, m.shape[1] - 1), np.clip(xi[ok] // q, 0, m.shape[2] - 1)]
        below[win[f]].append(float(np.median(y[ok] - top[ok])))
        cross[win[f]].append(float(np.mean((y[ok] > top[ok] + 2) & (cls != SEG_WATER) & (cls != SEG_SKY))))
    med = lambda v: np.array([np.median(x) if x else np.nan for x in v])
    return t0s, med(below), med(cross), win


def process_clip(root, clip, pose_paths, out_path, target_fps=8.0, focal_scale=1.0, rot_path=None, drop_still=False,
                 seg_path=None, variants=None):
    """Flow + water-plane speed for one clip. Returns a summary dict. With `rot_path` the rotations
    (derotation, up) come from that run (resampled to the DPVO frames) instead of DPVO.
    `variants` [(out_path, dict(mask=, horizon=, still=))] share one flow computation: `mask` keeps
    only PanSR water (seg_run.py file `seg_path`), `horizon` places the band and the water-plane
    geometry on the segmentation horizon instead of DPVO's, `still` = drop_still. Default: one
    variant (out_path, no segmentation, drop_still)."""
    t0 = time.time()
    frames, dec, times, K = clip_frames(root, clip, target_fps, focal_scale)
    t_dec = time.time() - t0
    z, pose_used = best_pose_run(pose_paths)
    if not np.array_equal(z['frame_indices'], dec):
        raise ValueError('DPVO frames differ from the decoded frames')
    R = quat_to_rotmat(np.nan_to_num(z['poses'][:, 3:7], nan=0.0) + np.where(
        np.isfinite(z['poses'][:, 3:7]).all(1, keepdims=True), 0, [0, 0, 0, 1]))
    pose_ok = np.isfinite(z['poses']).all(1)
    if rot_path is not None:
        R, pose_ok = resampled_rotations(rot_path, times)
        pose_used = f'{pose_used} (rotations: {rot_path})'
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
    if variants is None:
        variants = [(out_path, dict(mask=False, horizon=False, still=drop_still))]
    seg = dict(np.load(seg_path)) if seg_path is not None else None
    if seg is None and any(v['mask'] or v['horizon'] for _, v in variants):
        raise FileNotFoundError('segmentation file needed for this variant')
    segwater = seg_water_points(seg, dec, p) if seg is not None else None
    useg, lines = seg_horizon_ups(seg, times, K, W, H) if seg is not None else (None, None)
    hs = {}
    if seg is not None and any(v.get('hs') is not None for _, v in variants):
        dpvo_up = np.einsum('kij,kj->ki', np.stack([correction(a, b) for a, b in zip(hp, hr)]), ups)
        dpvo_up /= np.linalg.norm(dpvo_up, axis=1, keepdims=True)
        t0s, below, cross, win = horizon_sanity_stats(seg, dpvo_up, K, times, H, W)
        cand_w = (np.nan_to_num(below, nan=-np.inf) > HS_CAND_PX) | (np.nan_to_num(cross) > HS_CAND_CROSS)
        hs = dict(t0s=t0s, below=below, cross=cross, pair_window=win[:-1], cand=cand_w[win[:-1]])
        for u_px in {v['hs'] for _, v in variants if v.get('hs') is not None}:
            hs[u_px] = seg_horizon_ups(seg, times, K, W, H, shift_px=u_px)[0]
    usky, lines_sky = (seg_horizon_ups(seg, times, K, W, H, sky_only=True)
                       if seg is not None and any(v.get('skyhorizon') for _, v in variants) else (None, None))
    still_all = still_runs(flows, gy.ravel()) if any(v['still'] for _, v in variants) else None
    summary, done = None, {}
    for vpath, v in variants:
        uh, lh = (usky, lines_sky) if v.get('skyhorizon') else (useg, lines)
        base = done.get((v['mask'], v['still'])) if v.get('hs') is not None else None
        if v.get('hs') is not None and base is None:
            raise ValueError('an hsU variant needs the same variant without hs listed before it')
        summary, out = _fit_variant(vpath, v, clip, flows, p, rays, static, still_all, segwater, uh, lh, R, ups, hp, hr,
                                    htable, pose_ok, times, dec, K, focal_scale, n, pose_used, t0, t_dec, t_flow,
                                    hs if v.get('hs') is not None else None, base)
        if v.get('hs') is None and not (v['horizon'] or v.get('skyhorizon') or v.get('fixed')):
            done[(v['mask'], v['still'])] = out
    return summary


def _fit_variant(out_path, v, clip, flows, p, rays, static, still_all, segwater, useg, lines, R, ups, hp, hr, htable,
                 pose_ok, times, dec, K, focal_scale, n, pose_used, t0, t_dec, t_flow, hs=None, base=None):
    """Steps 3-5 for one variant of process_clip (see there); writes out_path, returns (summary, per-pair results).
    hsU variants (`hs` = horizon_sanity_stats tables + shifted segmentation ups, `base` = the same variant without
    hs): pairs in candidate windows use the segmentation horizon shifted up by U px, all others are copied from
    `base`; common.hs_merge then applies the fitted rule."""
    drop_still = v['still']
    still = still_all if drop_still else np.zeros((n - 1, len(p)), bool)
    n_water = n_removed = n_seg_drop = 0
    seg_h = (v['horizon'] or v.get('skyhorizon', False)) and useg is not None
    fixed = v.get('fixed', False)
    n_band = {b: np.zeros(n - 1, np.int32) for b in ('near', 'far')}
    dpvo_up = np.einsum('kij,kj->ki', np.stack([correction(a, b) for a, b in zip(hp, hr)]), ups)
    dpvo_up /= np.linalg.norm(dpvo_up, axis=1, keepdims=True)

    # 3-5. per pair: water pixels, near/far bands, robust fit
    keys = ('a_all', 'b_all', 'rmse_all', 'n_all', 'a_near', 'b_near', 'rmse_near', 'n_near',
            'a_far', 'b_far', 'rmse_far', 'n_far')
    out = {k: np.full(n - 1, np.nan) for k in keys}
    u_hs = hs[v['hs']] if hs is not None else None
    for k in range(n - 1):
        if hs is not None and not (hs['cand'][k] and u_hs is not None and np.isfinite(u_hs[k]).all()):
            for key in keys:
                out[key][k] = base[key][k]
            continue
        if not (pose_ok[k] and pose_ok[k + 1]):
            continue
        u = u_hs[k].copy() if hs is not None else useg[k] if seg_h else correction(hp[k], hr[k]) @ ups[k]
        u /= np.linalg.norm(u)
        water, s = water_region(rays, u, static)
        split = np.median(s[water]) if fixed and water.any() else None   # geometric bands, before masking
        if v['mask']:
            n_seg_drop += int((water & ~segwater[k]).sum())
            water &= segwater[k]
        n_water += int(water.sum())
        n_removed += int((water & still[k]).sum())
        water &= ~still[k]
        if water.sum() < (MIN_BAND_PIXELS if fixed else 2 * MIN_BAND_PIXELS):
            continue
        R_ji = R[k + 1].T @ R[k]
        fwd = np.array([0, 0, 1.0]) - u[2] * u      # camera forward, flattened
        e1 = fwd / np.linalg.norm(fwd)
        e2 = np.cross(e1, u)                        # right (camera +x side)
        if split is None:
            split = np.median(s[water])
        if fixed:
            n_band['near'][k], n_band['far'][k] = int((water & (s >= split)).sum()), int((water & (s < split)).sum())
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
             static_frac=float(static.mean()), K=np.array(K), focal_scale=np.float64(focal_scale),
             drop_still=drop_still, still_removed=np.int64(n_removed), water_samples=np.int64(n_water),
             still_points=np.int64(still.any(0).sum()), seg_mask=bool(v['mask']), seg_horizon=bool(seg_h),
             seg_dropped=np.int64(n_seg_drop), **({'seg_lines': lines} if lines is not None else {}),
             **({'seg_up': useg} if seg_h else {}), dpvo_up=dpvo_up,
             **({'n_band_near': n_band['near'], 'n_band_far': n_band['far']} if fixed else {}),
             **({'hs_shift_px': np.float64(v['hs']), 'hs_window_t0': hs['t0s'], 'hs_window_below': hs['below'],
                 'hs_window_cross': hs['cross'], 'hs_pair_window': hs['pair_window'], 'hs_cand': hs['cand']}
                if hs is not None else {}))
    return dict(key=clip['key'], camera_mode=clip['camera_mode'], n_pairs=n - 1, poses=pose_used,
                focal_scale=focal_scale,
                valid_frac=float(np.isfinite(speeds['speed_all']).mean()),
                still_removed=n_removed, water_samples=n_water, still_points=int(still.any(0).sum()),
                seg_mask=bool(v['mask']), seg_horizon=bool(seg_h), seg_dropped=n_seg_drop,
                seg_line_frac=float(np.isfinite(lines).all(1).mean()) if lines is not None else None,
                horizon_pitch_deg=[round(float(np.min(hp)), 2), round(float(np.max(hp)), 2)],
                horizon_roll_deg=[round(float(np.min(hr)), 2), round(float(np.max(hr)), 2)],
                static_frac=float(static.mean()),
                decode_s=round(t_dec, 1), flow_s=round(t_flow, 1), total_s=round(time.time() - t0, 1),
                **({'hs_cand_frac': float(hs['cand'].mean()),
                    'hs_max_below_px': float(np.nanmax(hs['below'])) if np.isfinite(hs['below']).any() else None}
                   if hs is not None else {})), out


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
    root, clip, pose_paths, out_path, focal_scale, rot_path, drop_still, seg_path, variants = args
    try:
        return process_clip(root, clip, pose_paths, out_path, focal_scale=focal_scale, rot_path=rot_path,
                            drop_still=drop_still, seg_path=seg_path, variants=variants)
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
    ap.add_argument('--rot-poses', type=Path,
                    help='take the rotations from this run dir instead (e.g. Pi3X), slerp-resampled to the DPVO frames')
    ap.add_argument('--drop-still', action='store_true',
                    help='drop water samples whose flow stays far below their row for >= STILL_RUN pairs (droplets)')
    ap.add_argument('--seg', type=Path, help='seg_run.py output dir (PanSR masks) for --variant')
    ap.add_argument('--variant', nargs='+', help="NAME=FLAGS with FLAGS from mask,horizon,skyhorizon,fixed,still,hsU (U = px) "
                                               "joined by '+': one flow computation, outputs <output>-NAME (needs --seg); "
                                               "an hsU variant follows the same variant without hsU")
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
    variants = None
    if args.variant:
        if args.seg is None:
            ap.error('--variant needs --seg')
        variants = []
        for spec in args.variant:
            name, flags = spec.split('=', 1)
            fl = set(flags.split('+')) - {''}
            hsu = [f for f in fl if f[:2] == 'hs' and f[2:].replace('.', '', 1).isdigit()]
            if fl - set(hsu) - {'mask', 'horizon', 'skyhorizon', 'fixed', 'still', 'none'} or len(hsu) > 1:
                ap.error(f'unknown variant flags in {spec}')
            variants.append((args.output.parent / f'{args.output.name}-{name}', dict(
                mask='mask' in fl, horizon='horizon' in fl, still='still' in fl, skyhorizon='skyhorizon' in fl,
                fixed='fixed' in fl, hs=float(hsu[0][2:]) if hsu else None)))
    jobs = []
    for c in clips:
        out = args.output / 'speed' / f"{c['key']}.npz"
        vouts = [(d / 'speed' / f"{c['key']}.npz", v) for d, v in variants] if variants else None
        if (all(o.is_file() for o, _ in vouts) if vouts else out.is_file()) and not args.overwrite:
            continue
        seg = args.seg / f"{c['key']}.npz" if args.seg else None
        if vouts and not seg.is_file():
            continue                                  # not segmented (yet)
        rot = args.rot_poses / 'poses' / f"{c['key']}.npz" if args.rot_poses else None
        if rot is not None and not rot.is_file():
            continue                                  # no rotation source for this clip
        jobs.append((root, c, [d / 'poses' / f"{c['key']}.npz" for d in args.poses], out, args.focal_scale, rot,
                     args.drop_still, seg, vouts))
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
