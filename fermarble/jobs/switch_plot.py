#!/usr/bin/env python3
"""DPVO vs VGGT heading on selected clips (no references needed, works on test).

Per clip: the three DPVO seeds, their circular median and VGGT's heading (turned since the start,
deg, sigma 1 s), plus the wrapped DPVO-median minus VGGT difference. Table: total and net turn of
the median and of VGGT, per-seed total turn and finite-pose fraction, and the 60 s divergence the
switch uses. Run from the repository root with the DPVO env python:
  switch_plot.py --dataset DATASET --poses DIR DIR DIR --vggt DIR --clips clip_098,... --output PNG
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (_run_heading, finite_fraction, heading_divergence_60s, load_manifest, load_pose_npz,  # noqa: E402
                    load_timestamps, multi_run_heading, select_clips, total_turn_deg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--poses', type=Path, nargs='+', required=True)
    ap.add_argument('--vggt', type=Path, required=True)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = args.dataset.resolve().parent
    clips = select_clips(load_manifest(args.dataset), 'all', args.clips)
    fig, axes = plt.subplots(len(clips), 2, figsize=(15, 3.2 * len(clips)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[3, 1.2]))
    print(f"{'clip':<9}{'mode':<9}{'div60':>7}{'med total':>10}{'VGGT total':>11}{'med net':>8}{'VGGT net':>9}"
          f"  seed totals / finite fraction")
    for (ax, ax2), c in zip(axes, clips):
        k = c['key']
        t = load_timestamps(root, c)
        runs, seeds = [], []
        for d in args.poses:
            z = load_pose_npz(d / 'poses' / f'{k}.npz')
            runs.append((z['poses'], z['times']))
            h, _ = _run_heading(z['poses'], z['times'], t, 1.0)
            seeds.append((h, finite_fraction(z['poses'])))
        h, _ = multi_run_heading(runs, t, 1.0)
        zv = load_pose_npz(args.vggt / 'poses' / f'{k}.npz')
        hv, _ = _run_heading(zv['poses'], zv['times'], t, 1.0)
        div = heading_divergence_60s(t, h, hv)
        tt = t - t[0]
        for i, (hs, fin) in enumerate(seeds):
            if hs is not None:
                ax.plot(tt, np.degrees(hs), lw=0.7, alpha=0.6, label=f'DPVO seed {i} ({100 * fin:.0f} % finite)')
        ax.plot(tt, np.degrees(h), 'k', lw=1.8, label='DPVO median')
        ax.plot(tt, np.degrees(hv), 'r', lw=1.8, label='VGGT')
        ax.set_title(f"{k} ({c['camera_mode']}): 60 s divergence {div:.1f} deg")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('turned since start (deg, clockwise +)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        d = np.degrees(np.angle(np.exp(1j * (h - hv))))
        ax2.plot(tt, d - np.median(d), 'k', lw=1)
        ax2.set_title('DPVO median - VGGT (deg, median removed)', fontsize=9)
        ax2.set_xlabel('time (s)')
        ax2.grid(alpha=0.3)
        st = ', '.join(f'{total_turn_deg(hs):.0f}/{fin:.2f}' if hs is not None else f'-/{fin:.2f}' for hs, fin in seeds)
        print(f"{k:<9}{c['camera_mode']:<9}{div:>7.1f}{total_turn_deg(h):>10.0f}{total_turn_deg(hv):>11.0f}"
              f"{np.degrees(h[-1] - h[0]):>8.0f}{np.degrees(hv[-1] - hv[0]):>9.0f}  {st}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print('written', args.output)


if __name__ == '__main__':
    main()
