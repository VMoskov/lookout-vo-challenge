#!/usr/bin/env python3
"""GT-free check of the test speed profiles that changed with the segmentation flow (variant c).

Per clip: old pipeline speed (kwide: original flow, v3a thresholds, or constant) vs new (flow-seg c,
c's re-fitted thresholds); the flow reliability (fraction of pairs whose near and far bands agree);
the band composition (samples PanSR labels non-water and dropped, droplet-like still samples
dropped); and plausibility flags: speed jumps (x2 within 1 s), near-zero stretches (< 0.25 x the
median for > 5 s) and correlation < 0.5 with the old pipeline speed or with the old raw flow.
Writes runs/seg-eval/test_speed_diag.png and .md. No references.
Gated bundles: --decisions <gated dir>/decisions.json --new-dir <gated dir> --params <gated dir>/speed_params.json
--output STEM checks the clips whose decision uses the new flow; plateau = a flat stretch > 5 s or a
gap-filled stretch (no reliable pair) > 10 s.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import filter_flow_speed, flow_reliable_speed, load_manifest, load_timestamps, select_clips  # noqa: E402

TEST = Path('lookout-vo-test-v0.1/release_manifest.json')
E = Path('runs/seg-eval')
CLIPS = 'clip_092,clip_095,clip_097,clip_098,clip_103,clip_107,clip_108,clip_109,clip_111'


def smooth_raw(d, t, win_s=4.0):
    """Raw flow speed (all water samples, no reliability filter), rolling median, at frame times, / median."""
    tm, v = d['t_mid'], np.asarray(d['speed_all'], float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 10:
        return None
    g = np.arange(tm[0], tm[-1], 0.5)
    med = np.array([np.nanmedian(v[ok][np.abs(tm[ok] - x) <= win_s / 2]) if (np.abs(tm[ok] - x) <= win_s / 2).any()
                    else np.nan for x in g])
    s = np.interp(t, g[np.isfinite(med)], med[np.isfinite(med)])
    return s / np.median(s)


def flat_run(t, x, eps=1e-6):
    """Longest stretch (s) over which x stays exactly constant (gap-filled plateaus)."""
    best, start = 0.0, 0
    for i in range(1, len(x) + 1):
        if i == len(x) or abs(x[i] - x[i - 1]) > eps:
            best = max(best, t[i - 1] - t[start])
            start = i
    return best


def flags(t, new, old_pipe, raw_old, gap=0.0):
    g = np.arange(t[0], t[-1], 1.0)
    s = np.interp(g, t, new)
    jumps = int((np.abs(np.diff(np.log(np.maximum(s, 1e-6)))) > np.log(2)).sum())
    low = s < 0.25 * np.median(s)
    run = best = 0
    for x in low:
        run = run + 1 if x else 0
        best = max(best, run)
    c_pipe = float(np.corrcoef(old_pipe, new)[0, 1]) if old_pipe is not None else np.nan
    c_raw = float(np.corrcoef(raw_old, new)[0, 1]) if raw_old is not None else np.nan
    f = []
    flat = flat_run(t, new)
    if flat > 5 or gap > 10:
        f.append(f'plateau (flat {flat:.0f} s, gap {gap:.0f} s)')
    if jumps:
        f.append(f'{jumps} jumps x2/s')
    if best > 5:
        f.append(f'near-zero {best} s')
    if np.isfinite(c_pipe) and c_pipe < 0.5:
        f.append(f'corr old pipeline {c_pipe:.2f}')
    if np.isfinite(c_raw) and c_raw < 0.5:
        f.append(f'corr old raw flow {c_raw:.2f}')
    return jumps, best, c_pipe, c_raw, f


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gated_flow import longest_gap
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decisions', type=Path, help='gated decisions.json: check the clips that use the new flow')
    ap.add_argument('--clips', help='comma-separated clips to check (overrides the default list)')
    ap.add_argument('--new-dir', type=Path, default=Path('runs/flow-seg-test-c'))
    ap.add_argument('--params', type=Path, default=E / 'params_c.json')
    ap.add_argument('--output', type=Path, default=E / 'test_speed_diag')
    args = ap.parse_args()
    v3a = json.loads(Path('runs/speed-tune3/params_v3a_no_hjump.json').read_text())
    pc = json.loads(args.params.read_text())
    keys = args.clips or CLIPS
    if args.decisions:
        dec = json.loads(args.decisions.read_text())
        keys = ','.join(k for k, d in dec.items() if d['speed_source'] != 'old')
    if not keys:
        Path(f'{args.output}.md').write_text('No test clip uses the new flow.\n')
        print('No test clip uses the new flow.')
        return
    clips = select_clips(load_manifest(TEST), 'all', keys)
    fig, axes = plt.subplots(len(clips), 1, figsize=(14, 2.4 * len(clips)), sharex=False, squeeze=False)
    axes = axes[:, 0]
    L = ['| clip | camera | old speed | new speed | reliable pairs old -> new | band non-water (PanSR) | still samples | '
         'corr old pipeline / old raw | flags |', '|---|---|---|---|---|---|---|---|---|']
    for ax, c in zip(axes, clips):
        k = c['key']
        t = load_timestamps(TEST.parent, c)
        do, dn = dict(np.load(f'runs/flow-test/speed/{k}.npz')), dict(np.load(args.new_dir / 'speed' / f'{k}.npz'))
        so, io = filter_flow_speed(do, t, v3a)
        sn, inn = filter_flow_speed(dn, t, pc)
        old = so if io['speed_fallback'] == 'none' else None
        new = sn if inn['speed_fallback'] == 'none' else np.ones(len(t))
        rel_o = flow_reliable_speed(do, v3a['agree_tol'], v3a['side_max'], v3a['drop_backward'])[2]
        rel_n = flow_reliable_speed(dn, pc['agree_tol'], pc['side_max'], pc['drop_backward'], pc.get('band_min'),
                                    pc['window_s'])[2]
        raw_o = smooth_raw(do, t)
        nw, sd, st = int(dn['water_samples']), int(dn.get('seg_dropped', 0)), int(dn.get('still_removed', 0))
        jumps, low, cp, cr, fl = flags(t, new, old, raw_o, longest_gap(dn, pc) if inn['speed_fallback'] == 'none' else 0.0)
        tt = t - t[0]
        if raw_o is not None:
            ax.plot(tt, raw_o, color='0.6', lw=0.8, label='old raw flow (all water samples, 4 s median)')
        ax.plot(tt, old if old is not None else np.ones(len(t)), 'b', lw=1.5,
                label='old pipeline speed' + ('' if old is not None else f" (constant: {io['speed_fallback']})"))
        ax.plot(tt, new, 'r', lw=1.5, label='new (seg c) speed' + ('' if inn['speed_fallback'] == 'none' else ' (constant)'))
        ax.set_ylim(0, max(3.0, float(np.nanpercentile(new, 99)) * 1.1))
        ax.set_title(f"{k} {c['camera_mode']}: reliable pairs {rel_o:.2f} -> {rel_n:.2f}; band non-water {sd / max(1, nw + sd):.0%}; "
                     f"flags: {', '.join(fl) or 'none'}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
        L.append(f"| {k[5:]} | {c['camera_mode']} | {'flow' if old is not None else 'const (' + io['speed_fallback'] + ')'} | "
                 f"{'flow' if inn['speed_fallback'] == 'none' else 'const'} | {rel_o:.2f} -> {rel_n:.2f} | {sd / max(1, nw + sd):.0%} | "
                 f"{st / max(1, nw):.1%} | {cp:.2f} / {cr:.2f} | {', '.join(fl) or '-'} |")
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(f'{args.output}.png', dpi=100)
    Path(f'{args.output}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
