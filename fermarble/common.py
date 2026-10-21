"""Shared fermarble core: starter imports, pose geometry and the heading + constant-speed model.

Nothing in this module reads reference trajectories. Only fermarble/oracle.py may do that.

Pose convention: DPVO's slam.terminate() returns camera-to-world poses as rows
`tx ty tz qx qy qz qw` (lietorch SE3 layout, quaternion scalar LAST); it inverts its internal
world-to-camera poses before returning. Camera axes are OpenCV: +x right, +y down, +z forward.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

FERMARBLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FERMARBLE_DIR.parent
STARTER_DIR = REPO_ROOT / 'starter'
DEFAULT_DATASET = REPO_ROOT / 'release' / 'release_manifest.json'
DEFAULT_DPVO_DIR = REPO_ROOT.parent / 'DPVO-859bbbf'  # pinned 859bbbf clone

if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))
# tracking.py resolves DPVO_DIR at import time, so default it before anyone imports tracking.
os.environ.setdefault('DPVO_DIR', str(DEFAULT_DPVO_DIR))

from public_dataset import (asset_path, load_manifest, load_timestamps, read_json,  # noqa: E402
                            select_clips, validate_prediction, write_json)

SMOOTH_SIGMA_S = 1.0
START_WINDOW_S = 3.0  # heading 0 = mean flattened camera forward over the first 3 s
GRID_HZ = 10.0
MIN_POSES = 10


# --------------------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------------------
def quat_to_rotmat(q):
    """(N,4) quaternions [qx,qy,qz,qw] -> (N,3,3) rotation matrices. Normalises its input."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
    ], -2)


def rotmat_to_quat(R):
    """(N,3,3) rotation matrices -> (N,4) quaternions [qx,qy,qz,qw] with qw >= 0."""
    R = np.asarray(R, float)
    out = np.empty((len(R), 4))
    for k, m in enumerate(R):
        tr = np.trace(m)
        if tr > 0:
            s = 2 * np.sqrt(tr + 1)
            q = [(m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s, s/4]
        else:
            i = int(np.argmax(np.diag(m)))
            j, l = (i + 1) % 3, (i + 2) % 3
            s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[l, l])
            q = [0.0] * 4
            q[i] = s / 4
            q[j] = (m[j, i] + m[i, j]) / s
            q[l] = (m[l, i] + m[i, l]) / s
            q[3] = (m[l, j] - m[j, l]) / s
        q = np.asarray(q)
        out[k] = q if q[3] >= 0 else -q
    return out


def invert_poses(poses):
    """Invert (N,7) SE3 poses [t, q]: (R, t) -> (R^T, -R^T t)."""
    R = quat_to_rotmat(poses[:, 3:7])
    Rt = np.transpose(R, (0, 2, 1))
    t = -np.einsum('nij,nj->ni', Rt, poses[:, :3])
    return np.concatenate([t, rotmat_to_quat(Rt)], axis=1)


# --------------------------------------------------------------------------------------
# Time-series helpers
# --------------------------------------------------------------------------------------
def gaussian_smooth_uniform(values, sigma_samples):
    """Gaussian filter along axis 0 of uniformly sampled data, reflect-padded at both ends."""
    values = np.asarray(values, float)
    if sigma_samples <= 0 or len(values) < 2:
        return values.copy()
    radius = int(np.ceil(4 * sigma_samples))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()
    pad = min(radius, len(values) - 1)
    squeeze = values.ndim == 1
    v = values[:, None] if squeeze else values
    padded = np.pad(v, ((pad, pad), (0, 0)), mode='reflect')
    if pad < radius:  # very short series: extend with edge values for the remaining kernel
        padded = np.pad(padded, ((radius - pad, radius - pad), (0, 0)), mode='edge')
    out = np.stack([np.convolve(padded[:, k], kernel, mode='valid') for k in range(v.shape[1])], 1)
    return out[:, 0] if squeeze else out


def smooth_in_time(times, values, sigma_s=SMOOTH_SIGMA_S, grid_hz=GRID_HZ):
    """Resample irregular samples to a uniform grid, Gaussian-smooth (sigma in seconds).

    Returns (grid_times, smoothed_values). Intended for unwrapped angles or positions.
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    n = max(2, int(np.ceil((times[-1] - times[0]) * grid_hz)) + 1)
    grid = np.linspace(times[0], times[-1], n)
    v = values[:, None] if values.ndim == 1 else values
    res = np.stack([np.interp(grid, times, v[:, k]) for k in range(v.shape[1])], 1)
    dt = (grid[-1] - grid[0]) / (n - 1) if n > 1 else 1.0
    res = gaussian_smooth_uniform(res, sigma_s / dt if dt > 0 else 0)
    return grid, (res[:, 0] if values.ndim == 1 else res)


def integrate_planar(frame_times, heading, speed):
    """Integrate heading (rad) and speed at every frame time into an (N,3) path with z = 0.

    Heading is a compass-style bearing: 0 along +y ("north" / the start direction), increasing
    clockwise seen from above, so 90 deg is +x ("east" / right). The path is right-handed (not
    mirrored): x = right, y = forward, z = up.

    Each interval uses the average of its endpoint velocity vectors (trapezoid rule).
    """
    frame_times = np.asarray(frame_times, float)
    vel = np.asarray(speed, float)[:, None] * np.column_stack([np.sin(heading), np.cos(heading)])
    dt = np.diff(frame_times)[:, None]
    steps = 0.5 * (vel[1:] + vel[:-1]) * dt
    xy = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return np.column_stack([xy, np.zeros(len(xy))])


def straight_line(frame_times):
    """Constant-velocity straight line at bearing 0 (+y): the always-valid fallback prediction."""
    t = np.asarray(frame_times, float)
    return np.column_stack([np.zeros(len(t)), t - t[0], np.zeros(len(t))])


# --------------------------------------------------------------------------------------
# DPVO poses -> heading
# --------------------------------------------------------------------------------------
def clean_poses(poses, times):
    """Keep finite poses with valid quaternions. Returns (poses, times) or (None, None)."""
    poses = np.asarray(poses, float)
    times = np.asarray(times, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) != len(times):
        return None, None
    ok = np.isfinite(poses).all(1) & np.isfinite(times)
    ok &= np.linalg.norm(poses[:, 3:7], axis=1) > 1e-6
    poses, times = poses[ok], times[ok]
    if len(poses) < MIN_POSES:
        return None, None
    keep = np.concatenate([[True], np.diff(times) > 0])  # strictly increasing times
    return poses[keep], times[keep]


def plane_basis(up):
    """Orthonormal (e1, e2) spanning the plane perpendicular to `up`, with e1 x e2 = up."""
    up = up / np.linalg.norm(up)
    helper = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = helper - (helper @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


def estimate_up(R_wc):
    """World 'up' = minus the mean camera +y (down) axis over the clip, normalised."""
    up = -R_wc[:, :, 1].mean(axis=0)
    norm = np.linalg.norm(up)
    if not np.isfinite(norm) or norm < 1e-3:
        return None
    return up / norm


def heading_frame(R_wc, times, start_window_s=START_WINDOW_S):
    """Horizontal reference frame for headings: (up, e1, e2), or None without a usable up.

    e1 (0 deg) is the mean flattened camera forward direction over the first `start_window_s`
    seconds of poses (a window, because DPVO's first frames are noisy). e2 = e1 x up points to
    the right of it, so headings atan2(v.e2, v.e1) are clockwise seen from above, like a compass
    bearing: positive = turned right since the start. Falls back to an arbitrary horizontal axis if the start window has no
    horizontal forward direction.
    """
    up = estimate_up(R_wc)
    if up is None:
        return None
    times = np.asarray(times, float)
    start = times <= times[0] + start_window_s
    fwd = R_wc[start, :, 2]
    flat = fwd - np.outer(fwd @ up, up)
    norms = np.linalg.norm(flat, axis=1)
    ok = norms > 0.1
    mean = (flat[ok] / norms[ok, None]).mean(axis=0) if ok.any() else np.zeros(3)
    if np.linalg.norm(mean) < 0.1:
        e1, _ = plane_basis(up)
    else:
        e1 = mean / np.linalg.norm(mean)
    return up, e1, np.cross(e1, up)


def pose_heading(poses, times, invert=False):
    """Per-pose unwrapped heading (rad) of the camera forward axis about the estimated up,
    relative to the start direction (see heading_frame); clockwise, positive = turned right.

    Returns (heading, info) or (None, reason) when the geometry is unusable.
    """
    if invert:
        poses = invert_poses(poses)
    R = quat_to_rotmat(poses[:, 3:7])
    frame = heading_frame(R, times)
    if frame is None:
        return None, 'camera y axes cancel out; no usable up vector'
    up, e1, e2 = frame
    fwd = R[:, :, 2]
    a, b = fwd @ e1, fwd @ e2
    horiz = np.hypot(a, b)
    good = horiz > 0.1  # forward axis within ~84 deg of the horizontal plane
    if good.sum() < MIN_POSES:
        return None, 'forward axis nearly parallel to up for most poses'
    heading = np.unwrap(np.arctan2(b, a))
    if not good.all():  # bridge near-vertical poses from their neighbours
        idx = np.arange(len(heading))
        heading = np.interp(idx, idx[good], heading[good])
    tilt = np.degrees(np.arccos(np.clip(np.abs(fwd @ up), 0, 1)))
    return heading, dict(up=up, e1=e1, e2=e2, mean_forward_elevation_deg=float(90 - tilt.mean()),
                         up_norm_before_normalise=float(np.linalg.norm(R[:, :, 1].mean(0))))


def motion_heading(poses, times, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """Heading of DPVO's own (smoothed) translation, projected on the plane perpendicular to up,
    in the same start-referenced, clockwise (compass-style) frame as pose_heading.

    Returns (grid_times, heading, planar_speed) with the speed in DPVO's arbitrary units.
    """
    if invert:
        poses = invert_poses(poses)
    R = quat_to_rotmat(poses[:, 3:7])
    frame = heading_frame(R, times)
    if frame is None:
        return None
    _, e1, e2 = frame
    planar = np.column_stack([poses[:, :3] @ e1, poses[:, :3] @ e2])
    grid, xy = smooth_in_time(times, planar, sigma_s)
    vel = np.gradient(xy, grid, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    moving = speed > 1e-3 * max(np.median(speed), 1e-12)
    if moving.sum() < 2:
        return None
    heading = np.unwrap(np.interp(grid, grid[moving], np.unwrap(heading[moving])))
    return grid, heading, speed


def forward_motion_cosine(poses, times, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """Pose-convention diagnostic: median cosine between DPVO's horizontal translation direction
    and the horizontal camera forward axis, over moving samples (speed above its median).

    Inverting poses mostly mirrors the yaw, and a 3-D Sim(3) undoes a mirror of a planar path,
    so heading scores barely tell the conventions apart. This does: with the right
    (camera-to-world) convention a forward-moving boat gives a value near +1.
    """
    if invert:
        poses = invert_poses(poses)
    R = quat_to_rotmat(poses[:, 3:7])
    up = estimate_up(R)
    if up is None:
        return float('nan')
    e1, e2 = plane_basis(up)
    planar = np.column_stack([poses[:, :3] @ e1, poses[:, :3] @ e2])
    fwd = np.column_stack([R[:, :, 2] @ e1, R[:, :, 2] @ e2])
    grid, xy = smooth_in_time(times, planar, sigma_s)
    _, f = smooth_in_time(times, fwd, sigma_s)
    vel = np.gradient(xy, grid, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > np.median(speed)
    if moving.sum() < 2:
        return float('nan')
    cos = (vel * f).sum(1) / (speed * np.linalg.norm(f, axis=1) + 1e-12)
    return float(np.median(cos[moving]))


def heading_at_frames(pose_times, heading, frame_times, sigma_s=SMOOTH_SIGMA_S):
    """Smooth unwrapped per-pose heading in time and interpolate it to every frame time.

    Frames before the first / after the last pose hold the end values.
    """
    grid, smooth = smooth_in_time(pose_times, heading, sigma_s)
    return np.interp(frame_times, grid, smooth)


def _run_heading(poses, pose_times, frame_times, sigma_s, invert=False):
    """One DPVO run -> (start-referenced heading at every frame, info) or (None, reason)."""
    poses, pose_times = clean_poses(poses, pose_times) if poses is not None else (None, None)
    if poses is None:
        return None, f'fewer than {MIN_POSES} finite poses'
    heading, info = pose_heading(poses, pose_times, invert=invert)
    if heading is None:
        return None, info
    h = heading_at_frames(pose_times, heading, frame_times, sigma_s)
    pose_cov = float((min(pose_times[-1], frame_times[-1]) - max(pose_times[0], frame_times[0]))
                     / (frame_times[-1] - frame_times[0]))
    return h, dict(n_poses=int(len(poses)), pose_time_coverage=pose_cov,
                   mean_forward_elevation_deg=info['mean_forward_elevation_deg'],
                   forward_motion_cos=forward_motion_cosine(poses, pose_times, sigma_s, invert))


def circular_median(headings):
    """Per-frame circular median of K heading series (K, N) in rad.

    Offsets from the first series are wrapped to (-pi, pi] before the median, so runs that
    differ by a full turn after unwrapping still agree; the result follows the first series'
    unwrapping. Exact for an odd K whenever the runs span less than 180 deg at a frame.
    """
    H = np.asarray(headings, float)
    offsets = np.angle(np.exp(1j * (H - H[0])))
    return H[0] + np.median(offsets, axis=0)


def _finish(frame_times, h, info):
    positions = integrate_planar(frame_times, h, np.ones(len(frame_times)))
    if not np.isfinite(positions).all():
        return straight_line(frame_times), dict(method='straight_line_fallback',
                                                reason='non-finite integrated path')
    return positions, dict(method='dpvo_heading_const_speed', **info,
                           heading_span_deg=float(np.degrees(h.max() - h.min())))


def heading_constant_speed(poses, pose_times, frame_times, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """The main model: DPVO camera heading + constant speed, at every released frame time.

    Returns (positions (N,3), info dict). Falls back to a straight line when the poses are
    unusable, so every clip always gets a valid prediction; info['method'] records which.
    """
    frame_times = np.asarray(frame_times, float)
    h, info = _run_heading(poses, pose_times, frame_times, sigma_s, invert)
    if h is None:
        return straight_line(frame_times), dict(method='straight_line_fallback', reason=info)
    return _finish(frame_times, h, info)


def multi_run_heading(runs, frame_times, sigma_s=SMOOTH_SIGMA_S):
    """Circular-median heading at every frame from several DPVO runs.
    Returns (heading or None, info) where info has n_runs, n_runs_used, run deviations."""
    frame_times = np.asarray(frame_times, float)
    heads, infos, reasons = [], [], []
    for poses, pose_times in runs:
        h, info = _run_heading(poses, pose_times, frame_times, sigma_s)
        if h is None:
            reasons.append(info)
        else:
            heads.append(h)
            infos.append(info)
    if not heads:
        return None, dict(n_runs=len(runs), reason='; '.join(map(str, reasons)) or 'no runs')
    h = circular_median(heads) if len(heads) > 1 else heads[0]
    spread = [float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (x - h))))))) for x in heads]
    return h, dict(infos[0], n_runs=len(runs), n_runs_used=len(heads), run_median_abs_dev_deg=spread)


def multi_run_heading_constant_speed(runs, frame_times, sigma_s=SMOOTH_SIGMA_S, speed=None):
    """Like heading_constant_speed, from several DPVO runs (e.g. seeds) of one clip.

    `runs` is a list of (poses, pose_times); unusable runs are skipped. Each run's
    start-referenced heading is computed separately, and the per-frame circular median is
    integrated. With one usable run this equals heading_constant_speed exactly. `speed`
    (relative speed at every frame, e.g. from filter_flow_speed) replaces the constant speed.
    """
    frame_times = np.asarray(frame_times, float)
    h, info = multi_run_heading(runs, frame_times, sigma_s)
    if h is None:
        return straight_line(frame_times), dict(method='straight_line_fallback', **info)
    if speed is None:
        return _finish(frame_times, h, info)
    positions = integrate_planar(frame_times, h, speed)
    if not np.isfinite(positions).all():
        return _finish(frame_times, h, dict(info, speed_reason='non-finite path with flow speed'))
    return positions, dict(method='dpvo_heading_flow_speed', **info,
                           heading_span_deg=float(np.degrees(h.max() - h.min())))


def rolling_median(times, values, window_s):
    """Centred rolling median over +-window_s/2 seconds (NaNs ignored; NaN where none)."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    lo = np.searchsorted(times, times - window_s / 2)
    hi = np.searchsorted(times, times + window_s / 2, side='right')
    for i in range(len(values)):
        w = values[lo[i]:hi[i]]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def flow_speed_profile(d, frame_times, band='all', window_s=2.0):
    """Minimal flow-speed profile at every frame time (diagnostics): rolling median of one band,
    NaN -> the clip median. Relative units (Sim(3) absorbs the scale)."""
    v = np.asarray(d[f'speed_{band}'], float)
    med = np.nanmedian(v) if np.isfinite(v).any() else 1.0
    sm = rolling_median(d['t_mid'], v, window_s)
    sm = np.where(np.isfinite(sm), sm, med)
    return np.interp(frame_times, d['t_mid'], sm)


def horizon_jump(d):
    """Largest change (deg) of flow_speed.py's per-window horizon pitch correction: a horizon
    fit that jumps between windows (shoreline vs. horizon, wrong DPVO attitude) is suspect."""
    hw = np.asarray(d['horizon_windows'], float) if 'horizon_windows' in d else np.zeros((0, 3))
    return float(np.abs(np.diff(hw[:, 1])).max()) if hw.ndim == 2 and len(hw) > 1 else 0.0


def flow_quality_features(d, agree_tol=0.6):
    """Per-clip quality signals of a flow_speed.py result, computable without ground truth:
    reliable_frac   pairs whose near/far bands agree within agree_tol (log)
    ratio_std       std of log(near/far) (bands normalised by their clip medians)
    side_frac       median |sideways| / |displacement| of the fit (boats mostly move forward)
    backward_frac   pairs whose fitted motion points backwards (a < 0)
    rmse_px         median fit residual (px)
    horizon_jump    largest change of the per-window horizon pitch correction (deg)
    horizon_spread  range of the per-window horizon pitch correction (deg)
    slow_frac       pairs slower than 0.3 x the clip median (all band)"""
    near, far = np.asarray(d['speed_near'], float), np.asarray(d['speed_far'], float)
    ok = np.isfinite(near) & np.isfinite(far) & (near > 0) & (far > 0)
    a, b = np.asarray(d['a_all'], float), np.asarray(d['b_all'], float)
    fin = np.isfinite(a) & np.isfinite(b) & (np.hypot(a, b) > 0)
    allsp = np.asarray(d['speed_all'], float)
    f = dict(valid_frac=float(fin.mean()))
    if ok.sum() >= 10:
        lr = np.log((near[ok] / np.median(near[ok])) / (far[ok] / np.median(far[ok])))
        f.update(reliable_frac=float((np.abs(lr) <= agree_tol).sum() / len(near)), ratio_std=float(lr.std()))
    else:
        f.update(reliable_frac=0.0, ratio_std=float('nan'))
    if fin.sum() >= 10:
        f.update(side_frac=float(np.median(np.abs(b[fin]) / np.hypot(a[fin], b[fin]))),
                 backward_frac=float((a[fin] < 0).mean()),
                 rmse_px=float(np.nanmedian(np.asarray(d['rmse_all'], float)[fin])),
                 slow_frac=float((allsp[fin] < 0.3 * np.median(allsp[fin])).mean()))
    else:
        f.update(side_frac=float('nan'), backward_frac=float('nan'), rmse_px=float('nan'), slow_frac=float('nan'))
    hw = np.asarray(d['horizon_windows'], float) if 'horizon_windows' in d else np.zeros((0, 3))
    if hw.ndim == 2 and len(hw) > 1:
        f.update(horizon_jump=float(np.abs(np.diff(hw[:, 1])).max()), horizon_spread=float(np.ptp(hw[:, 1])))
    else:
        f.update(horizon_jump=0.0, horizon_spread=0.0)
    return f


def heading_divergence_60s(t, h_a, h_b, window_s=60.0):
    """Largest change, within any window_s window, of the wrapped difference between two heading
    series (rad, at the same times), after removing its median: how far two heading sources
    drift apart over a minute (deg). Used by the DPVO/VGGT switch."""
    d = np.degrees(np.angle(np.exp(1j * (np.asarray(h_a) - np.asarray(h_b)))))
    d = d - np.median(d)
    j = np.searchsorted(t, np.asarray(t) + window_s)
    ok = j < len(t)
    return float(np.abs(d[j[ok]] - d[ok]).max()) if ok.any() else float('nan')


# DPVO/VGGT heading switch (fitted on wide train clips, oracle.py --combo-vggt): use VGGT's heading for a
# clip when it and DPVO's 3-seed median diverge by more than this within some 60 s window.
VGGT_SWITCH_DEG = 54.25


def total_turn_deg(h):
    """Total absolute heading change (deg) of a per-frame heading series (rad)."""
    return float(np.degrees(np.abs(np.angle(np.exp(1j * np.diff(np.asarray(h))))).sum()))


def finite_fraction(poses):
    """Fraction of pose rows that are entirely finite."""
    return float(np.isfinite(np.asarray(poses)).all(1).mean()) if len(poses) else 0.0


# Thresholds of filter_flow_speed. Defaults are placeholders; oracle.py --speed-tune fits them on
# TRAIN clips only and writes them to a JSON file that make_predictions.py --speed-params reads.
# (NO_LIMIT instead of inf so the parameters stay valid JSON.)
NO_LIMIT = 1e6
FLOW_FILTER_DEFAULTS = dict(agree_tol=0.5, window_s=2.0, clamp_lo=0.0, clamp_hi=NO_LIMIT,
                            min_reliable_frac=0.3, gate_stat='none', gate_min=0.0,
                            side_max=NO_LIMIT, drop_backward=False, horizon_jump_max=NO_LIMIT)


def speed_variation(speed):
    """Per-clip variation of a relative speed profile, for the flow gate: (CV, max/median)."""
    speed = np.asarray(speed, float)
    med = float(np.median(speed))
    return float(speed.std() / speed.mean()), float(speed.max() / med) if med > 0 else float('inf')


def flow_reliable_speed(d, agree_tol, side_max=NO_LIMIT, drop_backward=False):
    """Stage 1 of filter_flow_speed: normalised near/far bands, agreement test, geometric mean.
    A pair is also unreliable when its fitted motion is more than side_max sideways
    (|sideways| / |displacement|) or, with drop_backward, points backwards: a boat under way
    moves forwards, so these flag scenes the water-plane model does not explain (harbour
    clutter, docking, side-looking views).
    Returns (t_mid, speed or NaN where unreliable, reliable fraction, n usable pairs)."""
    near = np.asarray(d['speed_near'], float)
    far = np.asarray(d['speed_far'], float)
    t = np.asarray(d['t_mid'], float)
    ok = np.isfinite(near) & np.isfinite(far) & (near > 0) & (far > 0)
    if ok.sum() < 10:
        return t, np.full(len(t), np.nan), float(ok.mean()), int(ok.sum())
    nn = near / np.median(near[ok])
    ff = far / np.median(far[ok])
    with np.errstate(divide='ignore', invalid='ignore'):
        agree = ok & (np.abs(np.log(nn / ff)) <= agree_tol)
        if side_max < 1.0 or drop_backward:
            a, b = np.asarray(d['a_all'], float), np.asarray(d['b_all'], float)
            if side_max < 1.0:
                agree &= np.abs(b) <= side_max * np.hypot(a, b)
            if drop_backward:
                agree &= a > 0
    return t, np.where(agree, np.sqrt(nn * ff), np.nan), float(agree.mean()), int(agree.sum())


def flow_speed_from_reliable(t, v, frame_times, window_s, clamp_lo, clamp_hi, smoothed=None):
    """Stage 2: rolling median, fill gaps with the clip median, clamp, sample at frame times.
    `smoothed` may pass a precomputed rolling_median(t, v, window_s). Returns (speed, filled frac)."""
    sm = rolling_median(t, v, window_s) if smoothed is None else smoothed
    med = float(np.nanmedian(v))
    filled = ~np.isfinite(sm)
    sm = np.clip(np.where(filled, med, sm), clamp_lo * med, clamp_hi * med)
    return np.interp(frame_times, t, sm / med), float(filled.mean())


def filter_flow_speed(d, frame_times, params=None):
    """Relative speed at every frame from flow_speed.py's per-pair measurements.

    1. Each band (near, far) is normalised by its clip median; a pair is reliable when both
       bands exist and |log(near/far)| <= agree_tol. Its speed is their geometric mean.
    2. Rolling median over window_s seconds of the reliable pairs.
    3. Clamp to [clamp_lo, clamp_hi] x the clip median of the reliable speeds.
    4. Pairs with no reliable value in their window get the clip median. If fewer than
       min_reliable_frac of the pairs are reliable, the whole clip falls back to constant speed.
    5. Gate: if gate_stat is 'cv' or 'maxmed' and the profile's own variation (speed_variation)
       is below gate_min, the clip keeps constant speed (flow only where speed really varies).
    Returns (speed at frame_times, info). Units are relative (clip median = 1).
    """
    p = dict(FLOW_FILTER_DEFAULTS, **(params or {}))
    frame_times = np.asarray(frame_times, float)
    jump = horizon_jump(d)
    if jump > p['horizon_jump_max']:
        return np.ones(len(frame_times)), dict(speed_fallback='horizon', horizon_jump=jump, reliable_frac=np.nan,
                                               fallback_frac=1.0)
    t, v, rel_frac, n_rel = flow_reliable_speed(d, p['agree_tol'], p['side_max'], p['drop_backward'])
    if rel_frac < p['min_reliable_frac'] or n_rel < 10:
        return np.ones(len(frame_times)), dict(speed_fallback='clip', reliable_frac=rel_frac, fallback_frac=1.0)
    speed, filled = flow_speed_from_reliable(t, v, frame_times, p['window_s'], p['clamp_lo'], p['clamp_hi'])
    cv, maxmed = speed_variation(speed)
    info = dict(reliable_frac=rel_frac, fallback_frac=filled, flow_cv=cv, flow_maxmed=maxmed)
    if p['gate_stat'] != 'none' and {'cv': cv, 'maxmed': maxmed}[p['gate_stat']] < p['gate_min']:
        return np.ones(len(frame_times)), dict(info, speed_fallback='gate')
    return speed, dict(info, speed_fallback='none')


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def load_pose_npz(path):
    """Load a run_dpvo.py npz: dict with poses, frame_indices, times (and metadata)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prediction_dict(key, frame_times, positions):
    """Starter-format prediction covering EVERY released frame with its exact time."""
    return dict(key=key, frame_indices=list(range(len(frame_times))),
                times=[float(t) for t in frame_times],
                positions=np.asarray(positions, float).tolist())


def gap_aware_coverage(pred_times, ref_times, max_gap_s=1.0):
    """Fraction of reference times supported by a prediction sample or by interpolation
    between predicted samples no more than `max_gap_s` apart (RULES.md eligibility gate)."""
    pred_times = np.asarray(pred_times, float)
    ref_times = np.asarray(ref_times, float)
    j = np.searchsorted(pred_times, ref_times)
    exact = np.zeros(len(ref_times), bool)
    inside = (j > 0) & (j < len(pred_times))
    hit = j < len(pred_times)
    exact[hit] = np.isclose(pred_times[j[hit]], ref_times[hit], rtol=0, atol=1e-6)
    ok = exact.copy()
    jj = j[inside]
    ok[inside] |= (pred_times[jj] - pred_times[jj - 1]) <= max_gap_s
    return float(ok.mean())
