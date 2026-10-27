#!/usr/bin/env python3
"""GT-free inspection of the kwide_crop test trajectories (runs/crop-eval/test-kwide, = the 11.85 bundle).

Per test clip: Pi3X (kwide_crop's run, pi3x-test-cropmerged), VGGT (vggt-test) and DPVO (median of the 3 seeds)
heading, each smoothed with sigma 1 s and sampled at 2 Hz: total turn (sum |change|), net turn (end - start),
the ratios Pi3X / VGGT and Pi3X / DPVO; Pi3X's median window HFOV vs the nominal one; and the submitted path's
shape: self-crossings, longest same-direction turn (> 360 deg = loop / spiral) and zigzags (alternating turns
of > 20 deg each, per minute). Flags: Pi3X turns differently from both others (ratios both > 1.5 or both
< 0.67 with |difference| > 30 deg: focal / zoom suspect), Pi3X focal scale outside 0.8-1.25, loop / spiral,
zigzag (>= 2 reversals per minute), yaw rate > 10 deg/s over 2 s (sharp turn or camera pan; dev's submitted paths reach 17.5), self-crossing.
The submitted heading is shown in the pose convention (the map-frame course turns the opposite way). Writes runs/test-inspect/{table.md,table.csv,contact_sheet.png}.
Reads predictions and model runs only (no references).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import _run_heading, load_manifest, load_pose_npz, load_timestamps, multi_run_heading, select_clips  # noqa: E402

TEST = Path('lookout-vo-test-v0.1/release_manifest.json')
PRED = Path('runs/crop-eval/test-kwide')
PI3X, VGGT = Path('runs/pi3x-test-cropmerged'), Path('runs/vggt-test')
DPVO = [Path(f'runs/merged-test-seed{s}-clahe') for s in range(3)]
O = Path('runs/test-inspect')


def at2hz(t, h):
    g = np.arange(t[0], t[-1], 0.5)
    return g, np.interp(g, t, np.unwrap(h))


def turns(t, h):
    if h is None:
        return np.nan, np.nan
    _, x = at2hz(t, h)
    return float(np.degrees(np.abs(np.diff(x)).sum())), float(np.degrees(x[-1] - x[0]))


def run(path, t):
    if not path.is_file():
        return None
    z = load_pose_npz(path)
    return _run_heading(z['poses'], z['times'], t, 1.0)[0]


def crossings(p, step=10):
    q = p[::step, :2]
    a, b = q[:-1], q[1:]
    n = 0
    for i in range(len(a)):
        for j in range(i + 2, len(a)):
            d1, d2 = b[i] - a[i], b[j] - a[j]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                continue
            r = a[j] - a[i]
            s, u = (r[0] * d2[1] - r[1] * d2[0]) / den, (r[0] * d1[1] - r[1] * d1[0]) / den
            n += 0 < s < 1 and 0 < u < 1
    return n


def extrema(x, thr):
    """Heading turning points with a hysteresis of thr (rad): start, each reversal, end."""
    pts, trend, ext = [x[0]], 0, x[0]
    for v in x[1:]:
        if trend == 0:
            if abs(v - x[0]) > thr:
                trend, ext = int(np.sign(v - x[0])), v
        elif (v - ext) * trend > 0:
            ext = v
        elif (ext - v) * trend > thr:
            pts.append(ext)
            trend, ext = -trend, v
    return np.array(pts + [x[-1]])


def shape(t, h):
    """Largest one-way turn (deg) and reversals per minute of turns of > 20 deg (zigzags)."""
    g, x = at2hz(t, h)
    e = extrema(x, np.radians(20))
    return float(np.degrees(np.abs(np.diff(e)).max())), (len(e) - 2) / ((g[-1] - g[0]) / 60)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    O.mkdir(parents=True, exist_ok=True)
    clips = select_clips(load_manifest(TEST), 'test', None)
    methods = json.loads((PRED / 'prediction_manifest.json').read_text())['clips']
    rows = []
    fig, axes = plt.subplots(5, 8, figsize=(32, 20))
    for i, c in enumerate(clips):
        k, mode = c['key'], c['camera_mode']
        t = load_timestamps(TEST.parent, c)
        p = np.asarray(json.loads((PRED / 'predictions' / f'{k}.json').read_text())['positions'], float)
        hs = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
        hsub = -np.r_[hs[:1], hs]   # map-frame course -> the pose heading convention (opposite sense)
        hp, hv = run(PI3X / 'poses' / f'{k}.npz', t), run(VGGT / 'poses' / f'{k}.npz', t)
        hd = multi_run_heading([(z['poses'], z['times']) for z in (load_pose_npz(d / 'poses' / f'{k}.npz') for d in DPVO
                                if (d / 'poses' / f'{k}.npz').is_file())], t, 1.0)[0]
        (tp, np_), (tv, nv), (td, nd), (ts, ns) = turns(t, hp), turns(t, hv), turns(t, hd), turns(t, hsub)
        z = load_pose_npz(PI3X / 'poses' / f'{k}.npz')
        fov, nom = float(np.median(z['window_hfov_deg'])), float(z['nominal_hfov_deg'])
        fscale = np.tan(np.radians(nom) / 2) / np.tan(np.radians(fov) / 2)
        longest, zz = shape(t, hsub)
        g, x = at2hz(t, hsub)
        rate = float(np.degrees(np.abs(x[4:] - x[:-4])).max() / 2)   # deg/s over 2 s
        nx = crossings(p)
        rv, rd = tp / tv, tp / td
        fl = []
        if (rv > 1.5 and rd > 1.5 or rv < 0.67 and rd < 0.67) and min(abs(tp - tv), abs(tp - td)) > 30:
            fl.append('Pi3X turn differs from VGGT and DPVO')
        if not 0.8 <= fscale <= 1.25:
            fl.append(f'Pi3X focal x{fscale:.2f}')
        if longest > 360:
            fl.append(f'loop/spiral ({longest:.0f} deg one way)')
        if zz >= 2:
            fl.append(f'zigzag ({zz:.1f}/min)')
        if rate > 10:
            fl.append(f'fast turn {rate:.0f} deg/s (sharp turn or camera pan)')
        if nx:
            fl.append(f'{nx} self-crossing(s)')
        rows.append(dict(clip=k, mode=mode, minutes=round((t[-1] - t[0]) / 60, 1), source=methods[k]['method'].split('+')[-1],
                         pi3x_total=tp, vggt_total=tv, dpvo_total=td, submitted_total=ts, pi3x_net=np_, vggt_net=nv,
                         dpvo_net=nd, submitted_net=ns, pi3x_vggt=rv, pi3x_dpvo=rd, pi3x_hfov=fov, nominal_hfov=nom,
                         focal_scale=fscale, longest_one_way=longest, max_rate_deg_s=rate, zigzag_per_min=zz, crossings=nx, flags='; '.join(fl)))
        ax = axes[i // 4, 2 * (i % 4)]
        sc = ax.scatter(p[::5, 0], p[::5, 1], c=t[::5] - t[0], s=2, cmap='viridis')
        ax.plot(p[0, 0], p[0, 1], 'g^', ms=8)
        ax.plot(p[-1, 0], p[-1, 1], 'rs', ms=6)
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f"{k} {mode}: submitted path\n{'; '.join(fl) or 'no flags'}", fontsize=9,
                     color='crimson' if fl else 'black')
        ax.tick_params(labelsize=6)
        ax = axes[i // 4, 2 * (i % 4) + 1]
        tt = t - t[0]
        for h, lab, col in ((hd, 'DPVO', 'tab:blue'), (hv, 'VGGT', 'tab:orange'), (hp, 'Pi3X', 'tab:green'),
                            (hsub, 'submitted', 'k')):
            if h is not None:
                ax.plot(tt, np.degrees(np.unwrap(h) - np.unwrap(h)[0]), color=col, lw=1.6 if lab == 'submitted' else 1,
                        ls='--' if lab == 'submitted' else '-', label=lab)
        ax.set_title(f'heading (deg): total Pi3X {tp:.0f} / VGGT {tv:.0f} / DPVO {td:.0f}\n'
                     f'Pi3X HFOV {fov:.1f} vs nominal {nom:.1f} (x{fscale:.2f})', fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('kwide_crop test trajectories (11.85): path coloured by time (green start, red end) and headings '
                 '(start-referenced, sigma 1 s). GT-free.', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(O / 'contact_sheet.png', dpi=70)
    with open(O / 'table.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ['| clip | mode | min | source | total turn Pi3X / VGGT / DPVO / submitted | net turn Pi3X / VGGT / DPVO / submitted | '
         'Pi3X/VGGT | Pi3X/DPVO | Pi3X HFOV vs nominal (focal x) | one-way max | max yaw rate (2 s) | zigzag/min | crossings | flags |',
         '|' + '---|' * 14]
    for r in rows:
        L.append(f"| {r['clip'][5:]} | {r['mode']} | {r['minutes']} | {r['source']} | {r['pi3x_total']:.0f} / {r['vggt_total']:.0f} / "
                 f"{r['dpvo_total']:.0f} / {r['submitted_total']:.0f} | {r['pi3x_net']:.0f} / {r['vggt_net']:.0f} / {r['dpvo_net']:.0f} / "
                 f"{r['submitted_net']:.0f} | {r['pi3x_vggt']:.2f} | {r['pi3x_dpvo']:.2f} | {r['pi3x_hfov']:.1f} vs {r['nominal_hfov']:.1f} "
                 f"(x{r['focal_scale']:.2f}) | {r['longest_one_way']:.0f} | {r['max_rate_deg_s']:.0f} deg/s | {r['zigzag_per_min']:.1f} | {r['crossings']} | {r['flags'] or '-'} |")
    (O / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
