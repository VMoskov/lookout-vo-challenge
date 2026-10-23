#!/usr/bin/env python3
"""runs/status_seg.md for seg_pipeline.sh (DPVO env; reads only evaluator output, run files and test
predictions; never references): segmentation coverage, re-fitted flow thresholds, the dev table vs
kwide_crop, the per-test-clip flow acceptance and speed-profile change, the clip_104 horizon check
(runs/seg-eval/clip104_horizon.png), and the bundle decision (best of a/b/c, packed only if not worse
than kwide_crop on train and val)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import asset_path, estimate_up, filter_flow_speed, load_manifest, load_timestamps, quat_to_rotmat, select_clips  # noqa: E402
from flow_speed import best_pose_run, correction, horizon_rows  # noqa: E402

R = Path('runs')
E = R / 'seg-eval'
TEST = Path('lookout-vo-test-v0.1/release_manifest.json')
VARIANTS = {'refit_only': 'orig', 'a_mask': 'a', 'b_mask_horizon': 'b', 'c_mask_horizon_still': 'c'}
BUNDLE = R / 'submission_test_pi3x_ir_kwide_crop_seg.json'


def speed_series(flow_dir, key, t, params):
    p = Path(flow_dir) / 'speed' / f'{key}.npz'
    if not p.is_file():
        return None, 'no flow file'
    s, info = filter_flow_speed(np.load(p), t, params)
    return (s if info['speed_fallback'] == 'none' else None), info['speed_fallback']


def test_table():
    clips = [c for c in select_clips(load_manifest(TEST), 'test', None) if c['camera_mode'] in ('wide', 'flir-VIS')]
    base_params = json.loads(Path('runs/speed-tune3/params_v3a_no_hjump.json').read_text())
    dirs = {'kwide_crop': ('runs/flow-test', base_params)}
    for name, v in VARIANTS.items():
        pf = E / f'params_{v}.json'
        if pf.is_file():
            dirs[name] = ('runs/flow-test' if v == 'orig' else f'runs/flow-seg-test-{v}', json.loads(pf.read_text()))
    L = ['| clip | camera | ' + ' | '.join(dirs) + ' | speed change vs kwide_crop (b) |', '|---' * (len(dirs) + 3) + '|']
    for c in clips:
        t = load_timestamps(TEST.parent, c)
        cells, series = [], {}
        for name, (d, prm) in dirs.items():
            s, fb = speed_series(d, c['key'], t, prm)
            series[name] = s
            cells.append('**flow**' if fb == 'none' else f'const ({fb})')
        b, a = series.get('b_mask_horizon'), series.get('kwide_crop')
        if b is not None and a is not None:
            ch = f"corr {np.corrcoef(a, b)[0, 1]:.2f}, p90/p10 {np.percentile(a, 90) / np.percentile(a, 10):.2f} -> {np.percentile(b, 90) / np.percentile(b, 10):.2f}"
        elif b is not None:
            ch = f"const -> flow (p90/p10 {np.percentile(b, 90) / np.percentile(b, 10):.2f})"
        elif a is not None:
            ch = 'flow -> const'
        else:
            ch = 'const both'
        tag = ' **(bow/dome)**' if c['key'] in ('clip_093', 'clip_104', 'clip_108', 'clip_109') else ''
        L.append(f"| {c['key'][5:]}{tag} | {c['camera_mode']} | " + ' | '.join(cells) + f' | {ch} |')
    return L


def horizon_check():
    """clip_104: DPVO horizon (flow-test) vs segmentation horizon (flow-seg-test-b) vs PanSR water edge."""
    key = 'clip_104'
    fd, fs, sg = Path(f'runs/flow-test/speed/{key}.npz'), Path(f'runs/flow-seg-test-b/speed/{key}.npz'), Path(f'runs/seg-pansr/{key}.npz')
    if not (fd.is_file() and fs.is_file() and sg.is_file()):
        return ['(clip_104 inputs missing)']
    d, s, seg = np.load(fd), np.load(fs), np.load(sg)
    if 'seg_up' not in s.files:
        return ['clip_104: no segmentation horizon (too few frames with a water edge)']
    c = select_clips(load_manifest(TEST), 'all', key)[0]
    z, _ = best_pose_run([Path(f'runs/merged-test-seed{i}-clahe/poses/{key}.npz') for i in range(3)])
    Rm = quat_to_rotmat(np.where(np.isfinite(z['poses'][:, 3:7]), z['poses'][:, 3:7], [0, 0, 0, 1]))
    U = estimate_up(Rm[np.isfinite(z['poses']).all(1)])
    K = tuple(d['K'])
    from preprocessing import ClipPreprocessor   # starter
    t = load_timestamps(TEST.parent, c)
    pre = ClipPreprocessor(str(asset_path(TEST.parent, c['video'])), c['camera_mode'], 8.0, calibration=c['calibration'],
                           frame_timestamps=t)
    sf = seg['seg_frames']
    want = [int(sf[int(q * (len(sf) - 1))]) for q in (0.15, 0.5, 0.85)]
    tiles, rows = [], []
    for i, (_, img, _, _, _) in enumerate(pre.frames_with_time()):
        if i not in want:
            continue
        H, W = img.shape[:2]
        xs = np.arange(0, W, 2.0)
        u_d = correction(d['horizon_pitch_deg'][i], d['horizon_roll_deg'][i]) @ (Rm[i].T @ U)
        y_d = horizon_rows(u_d / np.linalg.norm(u_d), K, xs)
        y_s = horizon_rows(s['seg_up'][i], K, xs)
        top = seg['water_top'][list(sf).index(i)].astype(float)
        top[top < 0] = np.nan
        cx = int(W / 2)
        rows.append((float(seg['times'][i]), y_d[cx // 2], y_s[cx // 2], np.nanmedian(top[cx - 50:cx + 50])))
        im = img.copy()
        for y, col in ((y_d, (0, 0, 255)), (y_s, (0, 255, 0))):
            pts = np.column_stack([xs, y])[np.isfinite(y) & (y >= 0) & (y < H)].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(im, [pts.reshape(-1, 1, 2)], False, col, 2)
        tp = np.column_stack([np.arange(W), top])[np.isfinite(top)].astype(np.int32)
        for x, y in tp[::6]:
            cv2.circle(im, (int(x), int(y)), 1, (255, 255, 0), -1)
        cv2.putText(im, f'{key} t={seg["times"][i]:.0f}s  red DPVO horizon, green seg horizon, cyan PanSR water edge',
                    (8, 22), 0, 0.55, (255, 255, 255), 2)
        tiles.append(im)
    cv2.imwrite(str(E / 'clip104_horizon.png'), np.vstack(tiles))
    L = ['| t (s) | DPVO horizon row (centre) | seg horizon row | PanSR water edge row | DPVO - edge (px) | seg - edge (px) |',
         '|---|---|---|---|---|---|']
    for tt, yd, ys, ye in rows:
        L.append(f'| {tt:.0f} | {yd:.0f} | {ys:.0f} | {ye:.0f} | {yd - ye:+.0f} | {ys - ye:+.0f} |')
    L.append('\n(rows in the 960 px wide half-resolution frame; negative = above the water edge). Image: `runs/seg-eval/clip104_horizon.png`.')
    return L


def main():
    L = ['# Status: PanSR segmentation for flow', '',
         'Generated by `fermarble/jobs/status_seg.py` (job `fermarble/jobs/seg_pipeline.sh`, log `runs/seg_pipeline.log`). '
         'Nothing was uploaded.', '',
         'PanSR Swin-L (LaRS-pretrained, external pretraining) on every 2nd 8 fps flow frame (nearest mask in between), '
         'native half resolution. Variants (flow thresholds re-fitted on TRAIN for each, as v3a):',
         '- **refit_only**: original flow, re-fitted thresholds (control: separates the re-fit from the masking)',
         '- **a_mask**: DIS flow only on PanSR water inside the band',
         '- **b_mask_horizon**: a + horizon from the segmentation water edge (band placement and d = h / tan(alpha))',
         '- **c_mask_horizon_still**: b + --drop-still', '']
    segs = sorted(Path('runs/seg-pansr').glob('clip_*.npz'))
    L.append(f'Segmented clips: {len(segs)} (wide + flir-VIS, dev 75 + test 19 expected).')
    for v in ('orig', 'a', 'b', 'c'):
        pf = E / f'params_{v}.json'
        L.append(f"- thresholds {v}: {json.loads(pf.read_text()) if pf.is_file() else 'missing'}")
    L += ['', '## Dev vs kwide_crop', '']
    cmp_md, cmp_js = E / 'compare.md', E / 'compare.json'
    L += [cmp_md.read_text().strip()] if cmp_md.is_file() else ['(compare missing)']
    L += ['', '## Test: flow accepted or constant speed per clip', ''] + test_table()
    L += ['', '## clip_104 horizon', ''] + horizon_check()
    L += ['', '## Bundle', '']
    if cmp_js.is_file():
        v = json.loads(cmp_js.read_text())
        base = v.get('kwide_crop', {}).get('means')
        if base is None:   # compare_variants stores no means for the base row: recompute from its results
            r = {x['key']: x['drift_pct'] for x in json.loads(Path('runs/crop-eval/dev-kwide/results.json').read_text())}
            m = json.loads(Path('runs/crop-eval/dev-kwide/prediction_manifest.json').read_text())['clips']
            base = {s: float(np.mean([r[k] for k in r if m[k]['split'] == s])) for s in ('train', 'val')}
        ok = {n: x for n, x in v.items() if n in ('a_mask', 'b_mask_horizon', 'c_mask_horizon_still')
              and x['means']['train'] <= base['train'] + 1e-9 and x['means']['val'] <= base['val'] + 1e-9}
        if ok:
            best = min(ok, key=lambda n: ok[n]['means']['all dev'])
            if not BUNDLE.exists():
                res = subprocess.run([sys.executable, 'starter/pack_submission.py', '--dataset', str(TEST), '--split', 'test',
                                      '--predictions', str(E / f'test-{VARIANTS[best]}'), '--output', str(BUNDLE)],
                                     capture_output=True, text=True)
                L.append(f'Best variant not worse than kwide_crop on train and val: **{best}** -> `{BUNDLE}`. '
                         f'pack_submission: {res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-300:]}')
            else:
                L.append(f'Best variant: **{best}**; `{BUNDLE}` already exists (not overwritten).')
        else:
            L.append('No segmentation variant is at least as good as kwide_crop on both train and val: no bundle built.')
    (R / 'status_seg.md').write_text('\n'.join(L) + '\n')
    print('written', R / 'status_seg.md')


if __name__ == '__main__':
    main()
