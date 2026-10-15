#!/usr/bin/env python3
"""GPU step: run pretrained DPVO via starter tracking.track() and keep the FULL poses.

Per clip it writes
  <output>/poses/<clip>.npz           poses (N,7 camera-to-world tx ty tz qx qy qz qw),
                                      tracker_indices, frame_indices (decoded), times (released PTS)
  <output>/raw/predictions/<clip>.json  raw DPVO positions in starter format (plain baseline)
  <output>/run_manifest.json          one record per attempted clip, DPVO revision, environment

Resumable: a clip whose npz exists is skipped, and the npz is written atomically, so an
interrupted run is continued by rerunning the same command. Failed clips are recorded and
skipped on resume unless --retry-failed is given. Never reads references.
"""
from __future__ import annotations
import argparse
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # CPU-only self-test with a faked tracker
    torch = None

from common import (DEFAULT_DATASET, asset_path, load_manifest, load_timestamps, read_json, select_clips,
                    validate_prediction, write_json)


def save_npz_atomic(path: Path, **arrays):
    tmp = path.with_name(f'.{path.name}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--split', default='train,val', help='train, val, test, comma-separated, or all')
    ap.add_argument('--clips', help='comma-separated clip IDs; overrides --split')
    ap.add_argument('--output', required=True, type=Path, help='run directory (reused on resume)')
    ap.add_argument('--target-fps', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dpvo-buffer-size', type=int)
    ap.add_argument('--retry-failed', action='store_true', help='rerun clips recorded as failed')
    ap.add_argument('--clahe', action='store_true',
                    help='CLAHE-equalise thermal clips (flir-IR, infrared) before DPVO; see thermal.py')
    args = ap.parse_args(argv)

    from tracking import describe_environment, track  # imports DPVO lazily inside track()
    if args.clahe:
        import thermal
        thermal.install()  # track() now decodes through thermal.clahe_stream_worker
    if torch is not None:
        # The starter wraps DPVO's frame loop in torch.no_grad() but calls slam.terminate() after
        # it; terminate() runs 12 update() calls with autograd on, whose graph can exceed 14 GiB
        # (out of memory on a 16 GB GPU). Autograd changes no forward value, so turn it off for
        # the whole process; the starter files stay unchanged.
        torch.set_grad_enabled(False)

    manifest = load_manifest(args.dataset)
    selected = select_clips(manifest, args.split, args.clips)
    root = args.dataset.resolve().parent
    pose_dir = args.output / 'poses'
    raw_dir = args.output / 'raw' / 'predictions'
    pose_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / 'run_manifest.json'
    run = read_json(manifest_path) if manifest_path.is_file() else dict(clips={})
    settings = dict(target_fps=args.target_fps, seed=args.seed, dpvo_buffer_size=args.dpvo_buffer_size)
    if args.clahe:  # key only when on, so runs started before this option still resume
        settings['clahe'] = dict(modes=list(thermal.THERMAL_MODES), clip_limit=thermal.CLAHE_CLIP_LIMIT,
                                 tile_grid=list(thermal.CLAHE_TILE_GRID))
    if run.get('settings', settings) != settings:
        ap.error(f'--output was started with different settings {run["settings"]}; use a new directory')
    dpvo, environment = describe_environment(args.dpvo_buffer_size)
    run.update(settings=settings, dpvo=dpvo, environment=environment)
    print(f"DPVO revision={dpvo['revision']} checkpoint_sha256={dpvo['checkpoint_sha256']} "
          f"gpu={environment['gpu']} torch={environment['torch']}", flush=True)

    for number, clip in enumerate(selected, 1):
        key = clip['key']
        tag = f'[{number}/{len(selected)}] {key}'
        npz_path = pose_dir / f'{key}.npz'
        previous = run['clips'].get(key, {})
        if npz_path.is_file():
            print(f'{tag}: done, skipping', flush=True)
            continue
        if previous.get('status') == 'tracking_failed' and not args.retry_failed:
            print(f'{tag}: failed before, skipping (use --retry-failed)', flush=True)
            continue
        record = dict(split=clip['split'], camera_mode=clip['camera_mode'])
        if previous:  # keep the history of earlier (failed) attempts
            record['previous_attempts'] = previous.pop('previous_attempts', []) + [previous]
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print(f'{tag}: tracking ({clip["camera_mode"]})', flush=True)
        try:
            frame_times = load_timestamps(root, clip)
            video = asset_path(root, clip['video'])
            poses, tracker_idx, pre = track(str(video), clip['camera_mode'], args.target_fps,
                                            args.seed, calibration=clip['calibration'],
                                            frame_timestamps=frame_times,
                                            dpvo_buffer_size=args.dpvo_buffer_size)
            poses = np.asarray(poses, np.float64).reshape(-1, 7)
            tracker_idx = np.asarray(tracker_idx).astype(np.int64)
            frame_idx = np.array([pre.decoded_index(int(i)) for i in tracker_idx], np.int64)
            times = np.array([pre.frame_time(int(i)) for i in tracker_idx], np.float64)
            if not np.allclose(times, frame_times[frame_idx], rtol=0, atol=1e-6):
                raise RuntimeError('tracker times disagree with released frame timestamps')
            n_finite = int(np.isfinite(poses).all(1).sum())
            # Raw DPVO baseline, exactly as starter infer.py would write it.
            raw = dict(key=key, frame_indices=frame_idx.tolist(), times=times.tolist(),
                       positions=poses[:, :3].tolist())
            try:
                validate_prediction(raw, key, frame_times)
                write_json(raw_dir / f'{key}.json', raw)
                raw_status = 'written'
            except FloatingPointError as exc:
                raw_status = f'not_written: {exc}'
            save_npz_atomic(npz_path, poses=poses, tracker_indices=tracker_idx,
                            frame_indices=frame_idx, times=times,
                            n_frames=np.int64(len(frame_times)), native_fps=np.float64(pre.native_fps),
                            stride=np.int64(pre.stride), seed=np.int64(args.seed),
                            target_fps=np.float64(args.target_fps))
            record.update(status='tracked', n_poses=len(poses), n_finite_poses=n_finite,
                          raw_prediction=raw_status, native_fps=pre.native_fps, stride=pre.stride)
        except KeyboardInterrupt:
            print(f'{tag}: interrupted; rerun the same command to resume', flush=True)
            raise
        except Exception as exc:
            log = args.output / 'logs' / f'{key}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(traceback.format_exc())
            record.update(status='tracking_failed', reason=str(exc)[:300], error_log=str(log))
        record['elapsed_seconds'] = round(time.perf_counter() - started, 1)
        if torch is not None and torch.cuda.is_available():
            record['max_gpu_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            # Release a failed clip's tensors so one out-of-memory error does not cascade.
            gc.collect()
            torch.cuda.empty_cache()
        run['clips'][key] = record
        write_json(manifest_path, run)
        print(f"{tag}: {record['status']} ({record['elapsed_seconds']}s)"
              + (f" {record.get('n_finite_poses')}/{record.get('n_poses')} finite poses"
                 if record['status'] == 'tracked' else f" {record['reason']}"), flush=True)

    tracked = sum(1 for c in selected if (pose_dir / f"{c['key']}.npz").is_file())
    print(f'RUN_DPVO_COMPLETE: {tracked}/{len(selected)} clips have poses in {pose_dir}')
    return 0


if __name__ == '__main__':  # required: tracking uses 'spawn' multiprocessing
    raise SystemExit(main())
