#!/usr/bin/env python3
"""Gated segmentation flow (CPU, no references): per clip, choose between the old flow (as in kwide)
and the PanSR-masked flow, from GT-free features only. Gate parameters come from
oracle.py --gate-tune (fitted on TRAIN).

Inputs per clip: the old flow_speed.py file and three masked-flow files from flow_speed.py --variant
(fixed geometric near/far bands, droplet filter): 'm' (DPVO horizon), 'sa' (segmentation horizon,
top edge of water), 'sb' (segmentation horizon from the water/sky boundary only).
1. Horizon gate (per clip, so the distance scale h/tan(alpha) is consistent within a clip): use the
   segmentation horizon `hv` only if it differs from DPVO's by > T px (row at the image centre,
   median over the clip) or > D deg (tilt); else the masked flow with the DPVO horizon ('m').
2. Pixel selection: a band counts only with >= N PanSR water pixels (band_min in
   common.filter_flow_speed: one-band pairs with stricter reliability; none: constant speed).
3. Flow gate: the new flow replaces the old speed only if its reliable-pair fraction drops by <= X
   vs the old flow, it correlates >= Y with the old raw flow, and its longest stretch without
   reliable pairs (gap-filled) is < Z s; else the old speed (old flow or constant, as in kwide).
`build` writes a speed dir (one file per clip: the chosen flow), the matching filter parameters
(v3a + band_min) and the per-clip decisions.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (filter_flow_speed, flow_reliable_speed, load_manifest, load_timestamps, select_clips, write_json)
from flow_speed import horizon_rows

V3A = Path(__file__).resolve().parents[1] / 'runs/speed-tune3/params_v3a_no_hjump.json'
SOURCES = ('m', 'sa', 'sb')


def raw_profile(d, t, win_s=4.0):
    """Old raw flow speed (all water samples, no reliability filter): 4 s rolling median at frame
    times, divided by its median. None if too few samples."""
    tm, v = np.asarray(d['t_mid'], float), np.asarray(d['speed_all'], float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 10:
        return None
    tm, v = tm[ok], v[ok]
    g = np.arange(tm[0], tm[-1], 0.5)
    lo, hi = np.searchsorted(tm, g - win_s / 2), np.searchsorted(tm, g + win_s / 2)
    med = np.array([np.median(v[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    good = np.isfinite(med)
    s = np.interp(t, g[good], med[good])
    return s / np.median(s)


def horizon_diff(d):
    """(median |row difference| at the image centre, median |tilt difference| deg) between DPVO's
    and the segmentation horizon over the clip; (nan, nan) without a segmentation horizon."""
    if 'seg_up' not in d:
        return math.nan, math.nan
    K = tuple(d['K'])
    W = 2.0 * K[2]                              # processed frame width (principal point at the centre)
    xs = np.array([0.0, W / 2, W - 1.0])
    rows = lambda u: np.array([horizon_rows(x, K, xs) for x in u])
    idx = np.arange(0, len(d['dpvo_up']), 8)
    a, b = rows(d['dpvo_up'][idx]), rows(d['seg_up'][idx])
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if not ok.any():
        return math.nan, math.nan
    tilt = lambda r: np.degrees(np.arctan((r[:, 2] - r[:, 0]) / (W - 1)))
    return float(np.median(np.abs(a[ok, 1] - b[ok, 1]))), float(np.median(np.abs(tilt(a[ok]) - tilt(b[ok]))))


def longest_gap(d, params):
    """Longest stretch (s) without a reliable pair, including the clip ends (gap-filled stretch)."""
    t, v, _, _ = flow_reliable_speed(d, params['agree_tol'], params['side_max'], params['drop_backward'],
                                     params.get('band_min'), params['window_s'], params.get('near_w'),
                                     params.get('near_reliable', False))
    tr = t[np.isfinite(v)]
    if len(tr) == 0:
        return float(t[-1] - t[0])
    return float(max(np.diff(np.concatenate([[t[0]], tr, [t[-1]]])).max(), 0.0))


def clip_features(old_npz, new_npzs, frame_t, n_grid):
    """GT-free features of one clip for the gate. new_npzs: {source: path}."""
    v3a = json.loads(V3A.read_text())
    do = dict(np.load(old_npz))
    so, io = filter_flow_speed(do, frame_t, v3a)
    rel_old = flow_reliable_speed(do, v3a['agree_tol'], v3a['side_max'], v3a['drop_backward'])[2]
    raw_old = raw_profile(do, frame_t)
    f = dict(old_fallback=io['speed_fallback'], rel_old=rel_old, old_speed=so, raw_old=raw_old, src={})
    for src, path in new_npzs.items():
        if not Path(path).is_file():
            continue
        dn = dict(np.load(path))
        rd, td = horizon_diff(dn)
        per_n = {}
        for n in n_grid:
            prm = dict(v3a, band_min=n)
            sn, inn = filter_flow_speed(dn, frame_t, prm)
            rel = flow_reliable_speed(dn, prm['agree_tol'], prm['side_max'], prm['drop_backward'], n, prm['window_s'])[2]
            corr = float(np.corrcoef(raw_old, sn)[0, 1]) if raw_old is not None and inn['speed_fallback'] == 'none' else math.nan
            per_n[n] = dict(fallback=inn['speed_fallback'], speed=sn, rel=rel, corr=corr, gap=longest_gap(dn, prm))
        f['src'][src] = dict(row_diff=rd, tilt_diff=td, n=per_n)
    return f


def decide(f, g):
    """(source used or 'old', horizon source) for gate parameters g = dict(hv, N, T, D, X, Y, Z)."""
    hv = g['hv']
    s = f['src'].get(hv)
    use_seg_h = s is not None and ((np.isfinite(s['row_diff']) and s['row_diff'] > g['T'])
                                   or (np.isfinite(s['tilt_diff']) and s['tilt_diff'] > g['D']))
    src = hv if use_seg_h else 'm'
    if src not in f['src']:
        return 'old', src
    c = f['src'][src]['n'][g['N']]
    ok = (c['fallback'] == 'none' and f['rel_old'] - c['rel'] <= g['X'] and np.isfinite(c['corr'])
          and c['corr'] >= g['Y'] and c['gap'] < g['Z'])
    return (src if ok else 'old'), src


def build(args):
    g = json.loads(args.gate.read_text())
    root = args.dataset.resolve().parent
    clips = [c for c in select_clips(load_manifest(args.dataset), args.split, None) if c['camera_mode'] in ('wide', 'flir-VIS')]
    out = args.output / 'speed'
    out.mkdir(parents=True, exist_ok=True)
    decisions = {}
    for c in clips:
        k = c['key']
        t = load_timestamps(root, c)
        old = args.old / 'speed' / f'{k}.npz'
        new = {s: Path(f'{args.new_prefix}-{s}') / 'speed' / f'{k}.npz' for s in SOURCES}
        f = clip_features(old, new, t, [g['N']])
        choice, hsrc = decide(f, g)
        target = old if choice == 'old' else new[choice]
        link = out / f'{k}.npz'
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
        s = f['src'].get(hsrc, {})
        cn = s.get('n', {}).get(g['N'], {}) if s else {}
        decisions[k] = dict(camera_mode=c['camera_mode'], speed_source=choice, horizon=('seg' if hsrc != 'm' else 'dpvo'),
                            row_diff_px=s.get('row_diff'), tilt_diff_deg=s.get('tilt_diff'), old_fallback=f['old_fallback'],
                            rel_old=f['rel_old'], rel_new=cn.get('rel'), corr_raw=cn.get('corr'), gap_s=cn.get('gap'),
                            new_fallback=cn.get('fallback'))
    params = dict(json.loads(V3A.read_text()), band_min=g['N'])
    write_json(args.output / 'speed_params.json', params)
    nonan = lambda x: None if isinstance(x, float) and not math.isfinite(x) else x
    write_json(args.output / 'decisions.json', {k: {f: nonan(v) for f, v in d.items()} for k, d in decisions.items()})
    print(f"{sum(d['speed_source'] != 'old' for d in decisions.values())}/{len(decisions)} clips use the new flow; "
          f"{sum(d['horizon'] == 'seg' for d in decisions.values())} with the segmentation horizon")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--gate', type=Path, required=True, help='gate parameters JSON (oracle.py --gate-tune)')
    ap.add_argument('--old', type=Path, required=True, help='old flow dir (as in kwide)')
    ap.add_argument('--new-prefix', required=True, help='masked flow dirs <prefix>-m, -sa, -sb')
    ap.add_argument('--output', type=Path, required=True)
    build(ap.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
