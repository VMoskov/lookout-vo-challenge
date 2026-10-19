#!/usr/bin/env python3
"""Build runs/submission_test_flow_focal_v2.json = the focal bundle (v1) plus a ground-truth-free
x1.0 fallback: a flir-VIS test clip whose x2.0 DPVO runs ALL contain NaN poses (DPVO failed in
every seed) uses its x1.0 runs instead, if at least one x1.0 seed is fully finite. Only the fully
finite x1.0 seeds enter the median heading, and the x1.0 flow speed is used with them (its
geometry matches x1.0). Every other clip is exactly as in v1. Reads run manifests, not references.
Run from the repository root:  ~/micromamba/envs/dpvo/bin/python fermarble/jobs/build_focal_v2.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
TEST = 'lookout-vo-test-v0.1/release_manifest.json'
VIS = 'clip_093 clip_094 clip_095 clip_096 clip_098 clip_099 clip_104 clip_107 clip_108 clip_109 clip_111'.split()
OUT = Path('runs/submission_test_flow_focal_v2.json')
R = Path('runs')


def finite_runs(pattern):
    """{clip: [seed, ...]} of runs with no NaN pose, for runs/<pattern % seed>."""
    ok = {}
    for s in range(3):
        m = json.loads((R / (pattern % s) / 'run_manifest.json').read_text())['clips']
        for k, v in m.items():
            if v['status'] == 'tracked' and v['n_finite_poses'] == v['n_poses']:
                ok.setdefault(k, []).append(s)
    return ok


def main():
    if OUT.exists():
        sys.exit(f'{OUT} exists; not overwriting')
    ok2, ok1 = finite_runs('dpvo-test-fps8-seed%d-f2'), finite_runs('dpvo-test-fps8-seed%d')
    fallback = {k: ok1[k] for k in VIS if not ok2.get(k) and ok1.get(k)}
    print('x2.0 fully finite seeds per flir-VIS clip:', {k: ok2.get(k, []) for k in VIS})
    print('fallback to x1.0 (finite x1.0 seeds):', fallback)
    # pose dirs: v1 focal dirs, with fallback clips replaced by finite x1.0 seeds (others removed)
    for s in range(3):
        d = R / f'merged-test-seed{s}-focal-v2' / 'poses'
        shutil.rmtree(d.parent, ignore_errors=True)
        d.mkdir(parents=True)
        for f in sorted((R / f'merged-test-seed{s}-focal' / 'poses').glob('*.npz')):
            if f.stem in fallback:
                continue
            os.symlink(os.path.relpath(f.resolve(), d), d / f.name)
        for k, seeds in fallback.items():
            if s in seeds:
                src = R / f'merged-test-seed{s}-clahe' / 'poses' / f'{k}.npz'
                os.symlink(os.path.relpath(src.resolve(), d), d / f'{k}.npz')
    sp = R / 'flow-test-focal-v2' / 'speed'
    shutil.rmtree(sp.parent, ignore_errors=True)
    sp.mkdir(parents=True)
    for f in sorted((R / 'flow-test-focal' / 'speed').glob('*.npz')):
        src = (R / 'flow-test' / 'speed' / f.name) if f.stem in fallback else f
        os.symlink(os.path.relpath(src.resolve(), sp), sp / f.name)
    pred = R / 'heading-test-med3-flow-focal-v2'
    shutil.rmtree(pred, ignore_errors=True)
    subprocess.run([PY, 'fermarble/make_predictions.py', '--dataset', TEST, '--split', 'test',
                    '--poses', *[str(R / f'merged-test-seed{s}-focal-v2') for s in range(3)],
                    '--speed', str(sp.parent), '--speed-params', 'runs/speed-tune3/params_v3a_no_hjump.json',
                    '--speed-modes', 'wide,flir-VIS', '--output', str(pred)], check=True)
    subprocess.run([PY, 'starter/pack_submission.py', '--dataset', TEST, '--split', 'test',
                    '--predictions', str(pred), '--output', str(OUT)], check=True)
    sys.path.insert(0, 'starter')
    from public_dataset import load_manifest
    b = json.loads(OUT.read_text())
    v1 = json.loads(Path('runs/submission_test_flow_focal.json').read_text())
    pm = json.loads((pred / 'prediction_manifest.json').read_text())['clips']
    m = load_manifest(TEST)
    print(f'{OUT}: {len(b)} clips, all test clips: {sorted(b) == sorted(c["key"] for c in m["clips"])}')
    for c in m['clips']:
        k = c['key']
        i = pm[k]
        focal = 'x1.0 (fallback)' if k in fallback else ('x2.0' if k in VIS else 'x1.0')
        speed = 'FLOW' if i['method'] == 'dpvo_heading_flow_speed' else 'constant'
        same = b[k]['positions'] == v1[k]['positions']
        print(f"{k} {c['camera_mode']:<9} {focal:<16} runs {i.get('n_runs_used')}/{i.get('n_runs')}  speed {speed:<8}"
              f" ({i.get('speed_fallback', 'mode off')})  {'same as v1' if same else 'CHANGED vs v1'}")


if __name__ == '__main__':
    main()
