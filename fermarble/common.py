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

    Each interval uses the average of its endpoint velocity vectors (trapezoid rule).
    """
    frame_times = np.asarray(frame_times, float)
    vel = np.asarray(speed, float)[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
    dt = np.diff(frame_times)[:, None]
    steps = 0.5 * (vel[1:] + vel[:-1]) * dt
    xy = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return np.column_stack([xy, np.zeros(len(xy))])


def straight_line(frame_times):
    """Constant-velocity straight line along +x: the always-valid fallback prediction."""
    t = np.asarray(frame_times, float)
    return np.column_stack([t - t[0], np.zeros(len(t)), np.zeros(len(t))])


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


def pose_heading(poses, invert=False):
    """Per-pose unwrapped heading (rad) of the camera forward axis about the estimated up.

    Returns (heading, info) or (None, reason) when the geometry is unusable.
    """
    if invert:
        poses = invert_poses(poses)
    R = quat_to_rotmat(poses[:, 3:7])
    up = estimate_up(R)
    if up is None:
        return None, 'camera y axes cancel out; no usable up vector'
    e1, e2 = plane_basis(up)
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
    """Heading of DPVO's own (smoothed) translation, projected on the plane perpendicular to up.

    Returns (grid_times, heading, planar_speed) with the speed in DPVO's arbitrary units.
    """
    if invert:
        poses = invert_poses(poses)
    R = quat_to_rotmat(poses[:, 3:7])
    up = estimate_up(R)
    if up is None:
        return None
    e1, e2 = plane_basis(up)
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


def heading_constant_speed(poses, pose_times, frame_times, sigma_s=SMOOTH_SIGMA_S, invert=False):
    """The main model: DPVO camera heading + constant speed, at every released frame time.

    Returns (positions (N,3), info dict). Falls back to a straight line when the poses are
    unusable, so every clip always gets a valid prediction; info['method'] records which.
    """
    frame_times = np.asarray(frame_times, float)
    poses, pose_times = clean_poses(poses, pose_times) if poses is not None else (None, None)
    if poses is None:
        return straight_line(frame_times), dict(method='straight_line_fallback',
                                                reason=f'fewer than {MIN_POSES} finite poses')
    heading, info = pose_heading(poses, invert=invert)
    if heading is None:
        return straight_line(frame_times), dict(method='straight_line_fallback', reason=info)
    h = heading_at_frames(pose_times, heading, frame_times, sigma_s)
    positions = integrate_planar(frame_times, h, np.ones(len(frame_times)))
    if not np.isfinite(positions).all():
        return straight_line(frame_times), dict(method='straight_line_fallback',
                                                reason='non-finite integrated path')
    pose_cov = float((min(pose_times[-1], frame_times[-1]) - max(pose_times[0], frame_times[0]))
                     / (frame_times[-1] - frame_times[0]))
    return positions, dict(method='dpvo_heading_const_speed', n_poses=int(len(poses)),
                           pose_time_coverage=pose_cov,
                           mean_forward_elevation_deg=info['mean_forward_elevation_deg'],
                           forward_motion_cos=forward_motion_cosine(poses, pose_times, sigma_s, invert),
                           heading_span_deg=float(np.degrees(h.max() - h.min())))


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
