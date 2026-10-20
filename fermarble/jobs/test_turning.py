#!/usr/bin/env python3
"""Turning statistics of the DPVO heading on the 20 test clips (no references exist for test).

Per clip, from the 3-seed circular-median heading (smoothed, sigma 1 s) at every frame:
  total |heading change| (deg), max turn rate over 5 s windows (deg/s), full loops
  (heading span / 360), and per seed the total |heading change| and its spread across seeds.
Flags physically implausible turning or strong seed disagreement; for flagged clips also the
numbers at focal x2.0 (flir-VIS only) and a top-down plot of the path (median heading + constant
speed) at x1.0 and x2.0. Run from the repository root with the DPVO env python.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'fermarble')
from common import (_run_heading, integrate_planar, load_manifest, load_pose_npz, load_timestamps,
                    multi_run_heading)

TEST = Path('lookout-vo-test-v0.1/release_manifest.json')
OUT = Path('runs/test-turning')
X1 = [Path(f'runs/merged-test-seed{s}-clahe') for s in range(3)]
X2 = [Path(f'runs/dpvo-test-fps8-seed{s}-f2') for s in range(3)]
# Flag thresholds (deg, deg/s): a planing boat can sustain ~10-15 deg/s at most
MAX_RATE = 15.0
MAX_LOOPS = 2
SEED_SPREAD_DEG = 90.0
SEED_SPREAD_REL = 0.5


def stats(clip, dirs):
    root = TEST.parent
    t = load_timestamps(root, clip)
    runs, per_seed, nan = [], [], []
    for d in dirs:
        p = d / 'poses' / f"{clip['key']}.npz"
        if not p.is_file():
            continue
        z = load_pose_npz(p)
        runs.append((z['poses'], z['times']))
        nan.append(not np.isfinite(z['poses']).all())
        h, _ = _run_heading(z['poses'], z['times'], t, 1.0)
        per_seed.append(float(np.degrees(np.abs(np.diff(h)).sum())) if h is not None else np.nan)
    h, _ = multi_run_heading(runs, t, 1.0)
    if h is None:
        return None
    hd = np.degrees(h)
    j = np.searchsorted(t, t + 5.0)
    ok = j < len(t)
    rate = np.abs(hd[j[ok]] - hd[ok]) / (t[j[ok]] - t[ok])
    seeds = np.array(per_seed, float)
    med = np.nanmedian(seeds)
    return dict(total_deg=float(np.abs(np.diff(hd)).sum()), max_rate_dps=float(rate.max()),
                net_deg=float(hd[-1] - hd[0]), span_deg=float(np.ptp(hd)), loops=int(np.ptp(hd) // 360),
                seed_totals=[round(x, 0) for x in per_seed], seed_spread_deg=float(np.nanmax(seeds) - np.nanmin(seeds)),
                seed_spread_rel=float((np.nanmax(seeds) - np.nanmin(seeds)) / med) if med > 0 else np.nan,
                nan_seeds=int(sum(nan)), n_runs=len(runs)), (t, h)


def flags(s):
    f = []
    if s['max_rate_dps'] > MAX_RATE:
        f.append(f"turn rate {s['max_rate_dps']:.0f} deg/s")
    if s['loops'] >= MAX_LOOPS:
        f.append(f"{s['loops']} full loops")
    if s['seed_spread_deg'] > SEED_SPREAD_DEG and s['seed_spread_rel'] > SEED_SPREAD_REL:
        f.append(f"seeds disagree ({s['seed_spread_deg']:.0f} deg, {100 * s['seed_spread_rel']:.0f} %)")
    return f


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    clips = load_manifest(TEST)['clips']
    rows = []
    print(f"{'clip':<9}{'mode':<9}{'total |dH|':>11}{'net':>7}{'span':>7}{'loops':>6}{'max 5s rate':>12}"
          f"{'seed totals (deg)':>24}{'spread':>8}{'rel':>6}{'NaN seeds':>10}  flags")
    for c in clips:
        r = stats(c, X1)
        s, _ = r
        f = flags(s)
        rows.append(dict(key=c['key'], mode=c['camera_mode'], x1=s, flags=f))
        print(f"{c['key']:<9}{c['camera_mode']:<9}{s['total_deg']:>11.0f}{s['net_deg']:>7.0f}{s['span_deg']:>7.0f}{s['loops']:>6}"
              f"{s['max_rate_dps']:>12.1f}{str(s['seed_totals']):>24}{s['seed_spread_deg']:>8.0f}{s['seed_spread_rel']:>6.2f}"
              f"{s['nan_seeds']:>10}  {'; '.join(f) or '-'}")
    flagged = [r for r in rows if r['flags']]
    print(f"\nflagged: {[r['key'] for r in flagged] or 'none'}")
    for r in flagged:
        c = next(x for x in clips if x['key'] == r['key'])
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, dirs, col in (('x1.0', X1, 'tab:blue'), ('x2.0', X2, 'tab:red')):
            if not all((d / 'poses' / f"{c['key']}.npz").is_file() for d in dirs):
                continue
            s, (t, h) = stats(c, dirs)
            p = integrate_planar(t, h, np.ones(len(t)))
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.2, label=f"{label}: total {s['total_deg']:.0f} deg, "
                    f"max {s['max_rate_dps']:.1f} deg/s, seeds {s['seed_totals']}")
            ax.plot(*p[0, :2], 'o', color=col)
            if label == 'x2.0':
                r['x2'] = s
                print(f"  {c['key']} x2.0: total {s['total_deg']:.0f}, net {s['net_deg']:.0f}, span {s['span_deg']:.0f}, "
                      f"loops {s['loops']}, max rate {s['max_rate_dps']:.1f} deg/s, seeds {s['seed_totals']}, NaN seeds {s['nan_seeds']}")
        ax.set_aspect('equal')
        ax.set_title(f"{c['key']} ({c['camera_mode']}): median heading + constant speed (o = start)\n"
                     f"flags at x1.0: {'; '.join(r['flags'])}", fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"path_{c['key']}.png", dpi=100)
        plt.close(fig)
    (OUT / 'test_turning.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
