#!/usr/bin/env python3
"""Synthetic end-to-end self-test of the fermarble chain. No video, GPU, torch or DPVO needed.

Builds a fake release (manifest, timing, references; one test-split clip without reference).
It generates fake camera-to-world DPVO poses from a simulated boat with:
  - an arbitrary DPVO world rotation, scale and offset,
  - pitch/roll rocking,
  - a camera mounting offset (rotation and lever arm),
  - good orientation (small noise) but bad translation (scale drift and a random walk).
It then runs:
  run_dpvo.py with a fake tracking.track (interrupt + resume + a failed clip),
  make_predictions.py with the reference directory hidden,
  starter evaluate.py and starter pack_submission.py as subprocesses,
  oracle.py,
and checks the results. Exit code 0 means every check passed.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import common
from common import (STARTER_DIR, gap_aware_coverage, gaussian_smooth_uniform, heading_at_frames,
                    heading_constant_speed, integrate_planar, pose_heading, circular_median,
                    multi_run_heading_constant_speed,
                    invert_poses, quat_to_rotmat, rotmat_to_quat, write_json)

DURATION = 299.9
FAILURES = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


# --------------------------------------------------------------------------------------
# Synthetic world
# --------------------------------------------------------------------------------------
def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(rng):
    q = rng.normal(size=4)
    return quat_to_rotmat((q / np.linalg.norm(q))[None])[0]


class Boat:
    """Planar boat motion on a dense 100 Hz grid (ENU, metres, heading CCW from +x)."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.t = np.arange(0, DURATION + 1.0, 0.01)
        t = self.t
        yaw_rate = 0.02 * np.sin(2 * np.pi * t / rng.uniform(60, 120))
        for centre in rng.uniform(20, 280, 4):  # a few decisive turns
            yaw_rate += rng.choice([-1, 1]) * 0.08 * np.exp(-0.5 * ((t - centre) / 6) ** 2)
        speed = 5 + 2.5 * np.sin(2 * np.pi * t / rng.uniform(80, 160) + rng.uniform(0, 6))
        speed *= 1 - 0.95 * np.exp(-0.5 * ((t - rng.uniform(120, 180)) / 6) ** 4)  # near-stop
        self.yaw = rng.uniform(-np.pi, np.pi) + np.cumsum(yaw_rate) * 0.01
        vel = speed[:, None] * np.column_stack([np.cos(self.yaw), np.sin(self.yaw)])
        self.xy = np.vstack([[0, 0], np.cumsum(vel[:-1] * 0.01, axis=0)])
        self.pitch_amp, self.roll_amp = np.radians(2.0), np.radians(4.0)

    def position(self, times):
        return np.column_stack([np.interp(times, self.t, self.xy[:, 0]),
                                np.interp(times, self.t, self.xy[:, 1]), np.zeros(len(times))])

    def attitude(self, times):
        yaw = np.interp(times, self.t, self.yaw)
        pitch = self.pitch_amp * np.sin(2 * np.pi * times / 5.3)
        roll = self.roll_amp * np.sin(2 * np.pi * times / 7.1 + 1.0)
        return np.stack([rz(a) @ ry(b) @ rx(c) for a, b, c in zip(yaw, pitch, roll)])


# Boat body: x forward, y left, z up. OpenCV camera: x right, y down, z forward.
BOAT_FROM_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
MOUNT = rz(np.radians(7)) @ ry(np.radians(5)) @ rx(np.radians(2))  # yaw 7, pitch down 5, roll 2
LEVER_ARM = np.array([2.0, 0.5, 3.0])


def fake_dpvo_poses(boat, times, seed, world_rot=None, bad_translation=True):
    """Camera-to-world poses (N,7) in an arbitrary DPVO world, like slam.terminate()."""
    rng = np.random.default_rng(seed)
    R_wb = boat.attitude(times)
    R_wc = R_wb @ (MOUNT @ BOAT_FROM_CAM)
    p_cam = boat.position(times) + R_wb @ LEVER_ARM
    G = random_rotation(rng) if world_rot is None else world_rot
    noise = np.stack([rz(a) @ ry(b) @ rx(c) for a, b, c in rng.normal(0, np.radians(0.5), (len(times), 3))])
    R_d = G @ R_wc @ noise
    scale = 0.03 * (np.exp(np.linspace(0, np.log(3.0), len(times))) if bad_translation else 1.0)
    t_d = (scale[:, None] if bad_translation else scale) * (p_cam @ G.T) + np.array([4.0, -2.0, 7.0])
    if bad_translation:
        t_d += np.cumsum(rng.normal(0, 0.05, (len(times), 3)), axis=0)
    return np.concatenate([t_d, rotmat_to_quat(R_d)], axis=1)


def frame_times_for(fps, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, 1 / fps)
    t = t + rng.normal(0, 0.002, len(t))
    t[0] = 0.0
    t = np.maximum.accumulate(t)
    t[500] = t[499] + 1e-6  # a released-style ~1 microsecond interval
    for k in range(1, len(t)):
        if t[k] <= t[k - 1]:
            t[k] = t[k - 1] + 1e-6
    return t[t <= DURATION]


# Five clips: two good, one tracker failure, one all-NaN poses, one test clip without reference.
CLIPS = [
    dict(key='clip_000', split='val', mode='wide', fps=25.0, seed=1, poses='good'),
    dict(key='clip_001', split='train', mode='flir-IR', fps=30.0, seed=2, poses='good'),
    dict(key='clip_002', split='val', mode='wide', fps=24.0, seed=3, poses='fail'),
    dict(key='clip_003', split='train', mode='sionyx', fps=48.0, seed=4, poses='nan'),
    dict(key='clip_004', split='test', mode='wide', fps=25.0, seed=5, poses='good'),
]


def build_dataset(root: Path):
    clips, truth = [], {}
    for c in CLIPS:
        key = c['key']
        boat = Boat(c['seed'])
        ft = frame_times_for(c['fps'], c['seed'])
        entry = dict(key=key, split=c['split'], camera_mode=c['mode'],
                     calibration=dict(fx=900., fy=900., cx=480., cy=270., width=960, height=540,
                                      container_width=960, container_height=540,
                                      active_rect=[0, 0, 960, 540], provenance='nominal-fov-derived'),
                     video=f'videos/{key}.mp4', timestamps=f'timing/{key}.json',
                     reference=None if c['split'] == 'test' else f'references/{key}.json')
        clips.append(entry)
        write_json(root / entry['timestamps'], dict(key=key, times=ft.tolist()))
        if entry['reference']:
            rt = np.arange(0.0, np.floor(ft[-1]) + 1)
            if key == 'clip_000':
                rt = np.delete(rt, np.arange(100, 111))  # reference gap like clip_005
            pos = boat.position(rt)
            pos -= pos[0]
            write_json(root / entry['reference'], dict(key=key, times=rt.tolist(), positions=pos.tolist()))
        truth[key] = (boat, ft)
    counts = {s: sum(c['split'] == s for c in CLIPS) for s in ('train', 'val', 'test')}
    write_json(root / 'release_manifest.json', dict(schema_version=1, version='fermarble-selftest',
               counts={**counts, 'total': len(CLIPS)}, clips=clips))
    return truth


class FakePre:
    def __init__(self, frame_idx, frame_times, fps, stride):
        self._idx, self._t = frame_idx, frame_times
        self.native_fps, self.stride = fps, stride

    def decoded_index(self, i):
        return int(self._idx[i])

    def frame_time(self, i):
        return float(self._t[self._idx[i]])


def make_fake_track(truth, interrupt_on=None):
    calls = []

    def fake_track(video_path, preset, target_fps, seed, calibration=None, frame_timestamps=None,
                   dpvo_buffer_size=None):
        key = Path(video_path).stem
        calls.append(key)
        spec = next(c for c in CLIPS if c['key'] == key)
        if key == interrupt_on:
            raise KeyboardInterrupt
        if spec['poses'] == 'fail':
            raise RuntimeError('synthetic tracker failure')
        boat, ft = truth[key]
        assert np.array_equal(ft, frame_timestamps)
        stride = max(1, int(round(spec['fps'] / target_fps)))
        frame_idx = np.arange(stride - 1, len(ft), stride)  # what ClipPreprocessor yields
        poses = fake_dpvo_poses(boat, ft[frame_idx], spec['seed'] + 100)
        if spec['poses'] == 'nan':
            poses[:] = np.nan
        elif key == 'clip_001':
            poses[[5, 40, 41], :] = np.nan  # a few bad poses must be tolerated
        return poses, np.arange(len(frame_idx)), FakePre(frame_idx, ft, spec['fps'], stride)

    return fake_track, calls


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------
def multi_run_checks():
    print('multi-run (seed) heading combination')
    t = np.linspace(0, 10, 200)
    base = 0.3 * np.sin(t)
    med = circular_median([base, base + 0.1 + 2 * np.pi, base - 0.05])
    check(np.allclose(med, base), 'circular median ignores a full-turn unwrap offset between runs')
    med = circular_median([base, base + 0.02, base + 2.5])
    check(np.allclose(med, base + 0.02), 'circular median of 3 runs rejects one outlier run')
    boat = Boat(11)
    ft = np.arange(0, 200, 0.04)
    pt = ft[2::3]
    good1, good2 = fake_dpvo_poses(boat, pt, 21), fake_dpvo_poses(boat, pt, 22)
    bad = add_yaw_drift(fake_dpvo_poses(boat, pt, 23), pt, 40.0)
    p1, _ = heading_constant_speed(good1, pt, ft)
    m1, info1 = multi_run_heading_constant_speed([(good1, pt)], ft)
    check(np.array_equal(p1, m1) and info1['n_runs_used'] == 1, 'one run: identical to heading_constant_speed')
    m3, info3 = multi_run_heading_constant_speed([(good1, pt), (bad, pt), (good2, pt)], ft)
    truth = boat.position(ft)[:, :2]
    err = lambda p: np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)  # step lengths are all equal
    hd = lambda p: np.unwrap(np.arctan2(np.diff(p[:, 0]), np.diff(p[:, 1])))
    hg = hd(p1)
    dev3 = np.degrees(np.abs(np.angle(np.exp(1j * (hd(m3) - hg - np.median(hd(m3) - hg)))))).max()
    pbad, _ = heading_constant_speed(bad, pt, ft)
    devb = np.degrees(np.abs(np.angle(np.exp(1j * (hd(pbad) - hg - np.median(hd(pbad) - hg)))))).max()
    check(info3['n_runs_used'] == 3 and dev3 < 5 < devb,
          f'3 runs with one drifting: median heading stays with the good runs (max dev {dev3:.1f} vs {devb:.0f} deg)')
    _, info0 = multi_run_heading_constant_speed([(None, None), (np.full((50, 7), np.nan), pt[:50])], ft)
    check(info0['method'] == 'straight_line_fallback', 'no usable run -> straight-line fallback')


def unit_checks():
    print('unit checks')
    rng = np.random.default_rng(0)
    Rs = np.stack([random_rotation(rng) for _ in range(200)])
    check(np.allclose(quat_to_rotmat(rotmat_to_quat(Rs)), Rs, atol=1e-9), 'quaternion round trip')
    a = 0.7
    check(np.allclose(quat_to_rotmat(np.array([[0, 0, np.sin(a / 2), np.cos(a / 2)]]))[0], rz(a)),
          'quaternion is [qx,qy,qz,qw], scalar last')
    poses = np.concatenate([rng.normal(size=(50, 3)), rotmat_to_quat(Rs[:50])], 1)
    check(np.allclose(invert_poses(invert_poses(poses))[:, :3], poses[:, :3], atol=1e-9), 'pose inverse twice = identity')
    check(np.allclose(gaussian_smooth_uniform(np.full(30, 3.0), 4), 3.0), 'smoothing preserves constants')
    check(gap_aware_coverage(np.array([0, .5, 1, 3, 3.5]), np.array([0, .25, 1, 2, 3.5])) == 0.8,
          'gap-aware coverage rejects > 1 s gaps')
    # DPVO world rotation must not matter (up is estimated from the poses themselves)
    boat = Boat(7)
    ft = np.arange(0, 120, 0.04)
    pt = ft[2::3]
    p1, _ = heading_constant_speed(fake_dpvo_poses(boat, pt, 1, world_rot=np.eye(3)), pt, ft)
    p2, _ = heading_constant_speed(fake_dpvo_poses(boat, pt, 1, world_rot=random_rotation(rng)), pt, ft)
    d1, d2 = np.diff(p1[:, :2], axis=0), np.diff(p2[:, :2], axis=0)
    ang = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]) - np.arctan2(d2[:, 1], d2[:, 0]))
    check(np.ptp(ang) < np.radians(0.5), f'heading is DPVO-world invariant up to a constant (spread {np.degrees(np.ptp(ang)):.3f} deg)')
    # Heading convention: 0 = camera forward over the first 3 s, clockwise (right) positive.
    # The mount's 7 deg yaw offset is in both the start direction and every later heading, so
    # the camera heading must equal the boat's yaw change since the start.
    h, _ = pose_heading(fake_dpvo_poses(boat, pt, 2), pt)
    h = np.degrees(heading_at_frames(pt, h, pt))
    yaw = np.degrees(np.interp(pt, boat.t, boat.yaw))
    truth = -(yaw - yaw[pt <= pt[0] + 3.0].mean())  # boat yaw is CCW; heading is clockwise
    err = np.abs(h - truth)
    check(np.ptp(truth) > 30 and err.max() < 2.0,
          f'heading = yaw turned since start, right positive (turns span {np.ptp(truth):.0f} deg, max err {err.max():.2f} deg)')
    from oracle import true_motion
    bearing, _ = true_motion(np.arange(4.0), np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float))
    check(np.allclose(np.degrees(bearing), [0, 90, 180]), f'true heading is a compass bearing: N, E, S -> {np.degrees(bearing)}')
    p = integrate_planar(np.arange(3.0), np.radians([90.0, 90.0, 90.0]), np.ones(3))
    check(np.allclose(p[-1], [2, 0, 0]), 'bearing 90 deg integrates to +x (east / right)')
    check(abs(h[pt <= pt[0] + 3.0].mean()) < 1.0, f'heading averages ~0 over the first 3 s ({h[pt <= pt[0] + 3.0].mean():.2f} deg)')


def add_yaw_drift(poses, times, deg_per_min):
    """Rotate camera-to-world poses about the DPVO up axis by an angle growing linearly with time,
    so the clockwise-positive heading drifts by +deg_per_min (a DPVO yaw-drift simulation)."""
    up = common.estimate_up(quat_to_rotmat(poses[:, 3:7]))
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    out = poses.copy()
    for k, t in enumerate(times):
        a = -np.radians(deg_per_min * (t - times[0]) / 60.0)  # CCW about up = anticlockwise heading
        Rk = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        out[k, :3] = Rk @ poses[k, :3]
        out[k, 3:7] = rotmat_to_quat((Rk @ quat_to_rotmat(poses[k:k + 1, 3:7])[0])[None])[0]
    return out


def thermal_checks(work: Path):
    """CLAHE decoder: same frames/metadata as the starter worker for non-thermal clips, equalised
    frames for thermal ones; run_dpvo --clahe installs it. Uses a small synthetic video."""
    import queue
    import cv2
    import thermal
    import tracking
    print('thermal.py (CLAHE decoder)')
    path = work / 'synthetic_thermal.mp4'
    w, h, n = 320, 240, 40
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    rng = np.random.default_rng(0)
    for k in range(n):  # low-contrast grey scene with moving texture
        g = (120 + 8 * np.sin(np.arange(w)[None] / 9.0 + k / 3) + rng.normal(0, 2, (h, w))).clip(0, 255)
        vw.write(cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    times = np.arange(n) / 25.0
    cal = dict(fx=300., fy=300., cx=160., cy=120., width=w, height=h, container_width=w,
               container_height=h, active_rect=[0, 0, w, h], provenance='estimated')

    def frames(worker, preset):
        q = queue.Queue()
        worker(q, str(path), preset, 8.0, cal, times)
        out = []
        while True:
            item = q.get_nowait()
            if item[0] < 0:
                assert item[0] == -1, item[1]
                return out
            out.append(item)

    ref_wide, cl_wide = frames(tracking._stream_worker, 'wide'), frames(thermal.clahe_stream_worker, 'wide')
    ref_ir, cl_ir = frames(tracking._stream_worker, 'flir-IR'), frames(thermal.clahe_stream_worker, 'flir-IR')
    same_meta = all(a[0] == b[0] and a[3] == b[3] and a[4] == b[4] and np.array_equal(a[2], b[2])
                    for a, b in zip(ref_ir, cl_ir)) and len(ref_ir) == len(cl_ir) > 0
    check(len(ref_wide) == len(cl_wide) > 0 and all(np.array_equal(a[1], b[1]) for a, b in zip(ref_wide, cl_wide)),
          'CLAHE decoder leaves non-thermal (wide) frames byte-identical')
    check(same_meta, 'CLAHE decoder keeps indices, timestamps and intrinsics')
    std = lambda fr: np.mean([f[1].std() for f in fr])
    check(std(cl_ir) > 1.5 * std(ref_ir) and all(f[1].shape == g[1].shape and f[1].dtype == np.uint8
                                                   for f, g in zip(cl_ir, ref_ir)),
          f'CLAHE raises thermal frame contrast (std {std(ref_ir):.1f} -> {std(cl_ir):.1f}), same shape/dtype')
    colour = np.dstack([np.full((60, 80), 90, np.uint8), np.full((60, 80), 110, np.uint8),
                        (100 + 20 * np.sin(np.arange(80) / 5)).astype(np.uint8)[None].repeat(60, 0)])
    eq = thermal.apply_clahe(colour, cv2.createCLAHE(2.0, (8, 8)))
    check(eq.shape == colour.shape and not np.array_equal(eq[..., 0], eq[..., 2]),
          'colour frames are equalised on lightness and stay colour')


def run(cmd):
    res = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-2000:])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', type=Path, help='work in this new directory and keep it')
    args = ap.parse_args(argv)
    unit_checks()
    multi_run_checks()
    tmp = None
    if args.keep:
        work = args.keep
        work.mkdir(parents=True, exist_ok=False)
    else:
        tmp = tempfile.TemporaryDirectory(prefix='fermarble-selftest-')
        work = Path(tmp.name)
    try:
        thermal_checks(work)
        chain(work)
    finally:
        if tmp:
            tmp.cleanup()
    print(f'\nSELFTEST {"FAILED: " + str(len(FAILURES)) + " check(s)" if FAILURES else "PASSED"}')
    return 1 if FAILURES else 0


def chain(work: Path):
    data = work / 'release'
    manifest = data / 'release_manifest.json'
    truth = build_dataset(data)
    dpvo_run, pred_run, report = work / 'dpvo', work / 'pred', work / 'oracle'

    print('run_dpvo.py with a fake tracker')
    import run_dpvo
    import tracking
    real_track = tracking.track
    run_dpvo_main = lambda argv: run_dpvo.main([str(a) for a in argv])
    try:
        tracking.track, calls = make_fake_track(truth, interrupt_on='clip_001')
        try:
            run_dpvo_main(['--dataset', manifest, '--split', 'all', '--output', dpvo_run])
            check(False, 'interrupt propagates')
        except KeyboardInterrupt:
            check(True, 'interrupt propagates')
        check((dpvo_run / 'poses/clip_000.npz').is_file() and not list((dpvo_run / 'poses').glob('.*tmp')),
              'clip finished before the interrupt is saved, no temp files')
        tracking.track, calls = make_fake_track(truth)
        run_dpvo_main(['--dataset', manifest, '--split', 'all', '--output', dpvo_run])
        check('clip_000' not in calls and 'clip_001' in calls, f'resume skips done clips (tracked {calls})')
        tracking.track, calls = make_fake_track(truth)
        run_dpvo_main(['--dataset', manifest, '--split', 'all', '--output', dpvo_run])
        check(calls == [], 'second resume tracks nothing (failed clip skipped)')
        run_dpvo_main(['--dataset', manifest, '--split', 'all', '--output', dpvo_run, '--retry-failed'])
        check(calls == ['clip_002'], '--retry-failed retries only the failed clip')
        import thermal
        real_worker = tracking._stream_worker
        try:
            run_dpvo_main(['--dataset', manifest, '--clips', 'clip_001', '--output', work / 'dpvo_clahe', '--clahe'])
            rm_c = json.loads((work / 'dpvo_clahe/run_manifest.json').read_text())
            check(tracking._stream_worker is thermal.clahe_stream_worker and 'clahe' in rm_c['settings'],
                  'run_dpvo --clahe installs the CLAHE decoder and records it in the run settings')
        finally:
            tracking._stream_worker = real_worker
    finally:
        tracking.track = real_track
    rm = json.loads((dpvo_run / 'run_manifest.json').read_text())
    check(rm['clips']['clip_002']['status'] == 'tracking_failed', 'failed clip recorded')
    check(rm['clips']['clip_003']['raw_prediction'].startswith('not_written'), 'all-NaN raw prediction not written')
    z = np.load(dpvo_run / 'poses/clip_000.npz')
    ft0 = truth['clip_000'][1]
    check(z['poses'].shape[1] == 7 and np.allclose(z['times'], ft0[z['frame_indices']]), 'npz poses/indices/times consistent')
    check(run([STARTER_DIR / 'evaluate.py', '--dataset', manifest, '--split', 'train,val',
               '--predictions', dpvo_run / 'raw', '--output', work / 'raw_eval']).returncode == 0,
          'starter evaluate.py accepts raw DPVO predictions')

    print('make_predictions.py with references hidden')
    hidden = data / '.references_hidden'
    (data / 'references').rename(hidden)
    try:
        res = run([common.FERMARBLE_DIR / 'make_predictions.py', '--dataset', manifest, '--split', 'all',
                   '--poses', dpvo_run, '--output', pred_run])
    finally:
        hidden.rename(data / 'references')
    check(res.returncode == 0, 'make_predictions.py runs without any reference files')
    pm = json.loads((pred_run / 'prediction_manifest.json').read_text())['clips']
    check([pm[k]['method'] for k in sorted(pm)] == ['dpvo_heading_const_speed', 'dpvo_heading_const_speed',
          'straight_line_fallback', 'straight_line_fallback', 'dpvo_heading_const_speed'],
          'heading model for good clips, straight-line fallback for failed/NaN clips')
    for c in CLIPS:
        p = json.loads((pred_run / 'predictions' / f"{c['key']}.json").read_text())
        ft = truth[c['key']][1]
        xyz = np.asarray(p['positions'])
        ok = (p['frame_indices'] == list(range(len(ft))) and np.array_equal(np.asarray(p['times']), ft)
              and np.all(xyz[:, 2] == 0) and np.isfinite(xyz).all())
        check(ok, f"{c['key']}: every released frame, exact times, z = 0")

    print('starter evaluate.py on the predictions')
    res = run([STARTER_DIR / 'evaluate.py', '--dataset', manifest, '--split', 'train,val', '--predictions', pred_run])
    summary = json.loads((pred_run / 'summary.json').read_text())
    results = {r['key']: r for r in json.loads((pred_run / 'results.json').read_text())}
    check(res.returncode == 0 and summary['scored'] == summary['expected'] == 4, f"all 4 train/val clips scored {summary['by_status']}")
    check(all(r['temporal_coverage'] == 1.0 and r['max_prediction_gap_s'] < 1.0 for r in results.values()),
          'full overlap coverage and max gap < 1 s')

    print('starter pack_submission.py')
    for split in ('test', 'val'):
        out = work / f'submission_{split}.json'
        res = run([STARTER_DIR / 'pack_submission.py', '--dataset', manifest, '--split', split,
                   '--predictions', pred_run, '--output', out])
        n = len(json.loads(out.read_text())) if out.is_file() else 0
        check(res.returncode == 0 and n == sum(c['split'] == split for c in CLIPS), f'pack {split}: {n} clip(s)')
    res = run([STARTER_DIR / 'pack_submission.py', '--dataset', manifest, '--split', 'test',
               '--predictions', pred_run, '--output', work / 'submission_test.json'])
    check(res.returncode != 0, 'packer refuses to overwrite')

    print('heading difference (oracle.heading_difference)')
    import oracle
    clip0 = next(c for c in json.loads(manifest.read_text())['clips'] if c['key'] == 'clip_000')
    hd = oracle.heading_difference(data, clip0, dpvo_run / 'poses/clip_000.npz')
    check(hd is not None and hd[0]['label'] == 'constant' and abs(hd[0]['drift_deg_per_min']) < 1.0,
          f"no yaw drift -> label constant ({hd[0]['label']}, {hd[0]['drift_deg_per_min']:+.2f} deg/min)")
    z = dict(np.load(dpvo_run / 'poses/clip_000.npz'))
    z['poses'] = add_yaw_drift(z['poses'], z['times'], 6.0)
    (work / 'drift/poses').mkdir(parents=True)
    np.savez(work / 'drift/poses/clip_000.npz', **z)
    hd = oracle.heading_difference(data, clip0, work / 'drift/poses/clip_000.npz')
    check(hd is not None and hd[0]['label'] == 'drift' and abs(hd[0]['drift_deg_per_min'] - 6.0) < 1.0,
          f"injected +6 deg/min yaw drift recovered ({hd[0]['label']}, {hd[0]['drift_deg_per_min']:+.2f} deg/min)")

    print('oracle.py')
    res = run([common.FERMARBLE_DIR / 'make_predictions.py', '--dataset', manifest, '--split', 'all',
               '--poses', dpvo_run, work / 'drift', dpvo_run, '--output', work / 'pred_multi'])
    pm3 = json.loads((work / 'pred_multi/prediction_manifest.json').read_text())['clips']
    check(res.returncode == 0 and pm3['clip_000']['n_runs_used'] == 3 and pm3['clip_001']['n_runs_used'] == 2
          and pm3['clip_002']['method'] == 'straight_line_fallback',
          'make_predictions with several --poses dirs combines the available runs per clip')
    res = run([common.FERMARBLE_DIR / 'oracle.py', '--dataset', manifest, '--split', 'train,val',
               '--poses', dpvo_run, '--predictions', pred_run, '--output', report, '--heading-diff',
               '--worst', '3'])
    check(res.returncode == 0, 'oracle.py runs')
    check((report / 'heading_diff.csv').is_file() and (report / 'heading_diff_overview.png').is_file()
          and len(list((report / 'heading_diff').glob('*.png'))) == 2,
          'heading-diff CSV, overview and per-clip plots written (2 clips with poses)')
    print(res.stdout.split('\n\n', 1)[-1] if res.returncode == 0 else '')
    rows = {r['key']: r for r in csv.DictReader(open(report / 'per_clip.csv'))}
    f = lambda k, v: float(rows[k][v]) if rows[k][v] not in ('', 'nan') else float('nan')
    for k in rows:
        check(f(k, 'true_head_true_speed') < 0.3, f"{k}: true heading + true speed ~0 ({f(k, 'true_head_true_speed'):.3f}%)")
        check(abs(f(k, 'submitted_drift') - f(k, 'dpvo_head_const')) < 1e-3,
              f"{k}: evaluate.py drift == oracle dpvo_head_const ({f(k, 'submitted_drift'):.3f})")
        check(f(k, 'pipeline_gap_cov') == 1.0, f'{k}: gap-aware coverage 100%')
    for k in ('clip_000', 'clip_001'):
        dh, th = f(k, 'dpvo_head_const'), f(k, 'true_head_const')
        check(abs(dh - th) < 1.0, f'{k}: DPVO heading ~ true heading under constant speed ({dh:.2f} vs {th:.2f})')
        check(f(k, 'dpvo_head_true_speed') < 1.5, f"{k}: DPVO heading + true speed small ({f(k, 'dpvo_head_true_speed'):.2f})")
        check(dh < f(k, 'straight'), f"{k}: beats straight line ({f(k, 'straight'):.1f})")
        check(f(k, 'fwd_cos') > 0.9 and f(k, 'fwd_cos_inv') < f(k, 'fwd_cos') - 0.3,
              f"{k}: convention check, translation along camera +z ({f(k, 'fwd_cos'):.2f} vs inverted {f(k, 'fwd_cos_inv'):.2f})")
    check(f('clip_000', 'dpvo_head_const') < f('clip_000', 'raw_dpvo'),
          f"clip_000: beats raw DPVO with bad translation ({f('clip_000', 'raw_dpvo'):.1f})")
    check(np.isnan(f('clip_001', 'raw_dpvo')), 'clip_001: raw DPVO with NaN poses is not scored, as in the starter')
    for k in ('clip_002', 'clip_003'):
        check(abs(f(k, 'dpvo_head_const') - f(k, 'straight')) < 1e-6, f'{k}: fallback scores as straight line')


if __name__ == '__main__':
    raise SystemExit(main())
