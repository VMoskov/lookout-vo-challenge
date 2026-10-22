#!/usr/bin/env python3
"""GT-free DPVO tracking check per clip and focal scale (flir-IR, CLAHE): does DPVO track?

Per clip x focal: per-seed total |heading change| (deg), net turn and finite-pose fraction; the
3-seed median's total and net turn; seed agreement = mean pairwise heading_change_rms_60s between
seeds (deg; low = seeds agree); and the median's heading_change_rms_60s to VGGT and to Pi3X (deg;
low = same heading shape). No references. Writes CSV + prints a table.

  flirir_focal_check.py --dataset DS --clips c1,c2 --runs 1.0=DIR+DIR+DIR 0.5=... --vggt DIR --pi3x DIR --output CSV
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (_run_heading, finite_fraction, heading_change_rms_60s, load_manifest, load_pose_npz,  # noqa: E402
                    load_timestamps, multi_run_heading, select_clips, total_turn_deg)


def heading(path, t):
    if not path.is_file():
        return None, 0.0, None
    z = load_pose_npz(path)
    return _run_heading(z['poses'], z['times'], t, 1.0)[0], finite_fraction(z['poses']), (z['poses'], z['times'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='FOCAL=DIR+DIR+DIR')
    ap.add_argument('--vggt', type=Path)
    ap.add_argument('--pi3x', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.dataset.resolve().parent
    runs = {kv.split('=', 1)[0]: [Path(d) for d in kv.split('=', 1)[1].split('+')] for kv in args.runs}
    rows = []
    for c in select_clips(load_manifest(args.dataset), 'all', args.clips):
        k = c['key']
        t = load_timestamps(root, c)
        ref = {}
        for name, d in (('vggt', args.vggt), ('pi3x', args.pi3x)):
            if d is not None:
                ref[name] = heading(d / 'poses' / f'{k}.npz', t)[0]
        for focal, dirs in runs.items():
            seeds = [heading(d / 'poses' / f'{k}.npz', t) for d in dirs]
            hs = [h for h, _, _ in seeds if h is not None]
            h, _ = multi_run_heading([r for _, _, r in seeds if r is not None], t, 1.0)
            row = dict(key=k, focal=focal,
                       seed_total=[round(total_turn_deg(x)) if x is not None else None for x, _, _ in seeds],
                       seed_net=[round(float(np.degrees(x[-1] - x[0]))) if x is not None else None for x, _, _ in seeds],
                       seed_finite=[round(f, 2) for _, f, _ in seeds],
                       med_total=total_turn_deg(h) if h is not None else np.nan,
                       med_net=float(np.degrees(h[-1] - h[0])) if h is not None else np.nan,
                       seed_agree=float(np.mean([heading_change_rms_60s(t, a, b) for a, b in itertools.combinations(hs, 2)]))
                       if len(hs) > 1 else np.nan)
            for name, hr in ref.items():
                row[f'rms_{name}'] = heading_change_rms_60s(t, h, hr) if h is not None and hr is not None else np.nan
            rows.append(row)
        for name, hr in ref.items():
            if hr is not None:
                print(f"{k} {name}: total {total_turn_deg(hr):.0f} net {np.degrees(hr[-1] - hr[0]):+.0f}")
        if 'vggt' in ref and 'pi3x' in ref and ref['vggt'] is not None and ref['pi3x'] is not None:
            print(f"{k} VGGT vs Pi3X rms60 {heading_change_rms_60s(t, ref['vggt'], ref['pi3x']):.1f}")
    print(f"{'clip':<9}{'focal':>6}{'seed totals':>18}{'seed nets':>18}{'finite':>18}{'med tot':>8}{'med net':>8}"
          f"{'seed agree':>11}{'rms VGGT':>9}{'rms Pi3X':>9}")
    for r in rows:
        print(f"{r['key']:<9}{r['focal']:>6}{str(r['seed_total']):>18}{str(r['seed_net']):>18}{str(r['seed_finite']):>18}"
              f"{r['med_total']:>8.0f}{r['med_net']:>+8.0f}{r['seed_agree']:>11.1f}{r.get('rms_vggt', np.nan):>9.1f}"
              f"{r.get('rms_pi3x', np.nan):>9.1f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
