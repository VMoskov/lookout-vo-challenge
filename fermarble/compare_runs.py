#!/usr/bin/env python3
"""Compare evaluated prediction runs side by side (mean / median drift %, per group).

Each run is LABEL=DIR, where DIR holds starter evaluate.py's results.json. Groups: all clips,
all except zoom (zoom is ignored when choosing settings), each split, each camera mode. With
--clips-of MODE[,MODE] it also lists those clips one by one. Reads evaluator outputs and the
manifest only, never references.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

from common import DEFAULT_DATASET, load_manifest, read_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='LABEL=DIR (DIR contains results.json); the first is the baseline')
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--clips-of', help='also list per-clip drift for these camera modes (comma-separated)')
    ap.add_argument('--csv', type=Path, help='write the group table to this CSV')
    ap.add_argument('--exclude', help='comma-separated clip IDs to leave out (e.g. clips used for design)')
    ap.add_argument('--paired', action='store_true',
                    help='also count clips better/worse (> 0.5 points) than the first run, per group')
    args = ap.parse_args(argv)
    mode = {c['key']: c['camera_mode'] for c in load_manifest(args.dataset)['clips']}
    split = {c['key']: c['split'] for c in load_manifest(args.dataset)['clips']}
    runs = {}
    for spec in args.runs:
        label, path = spec.split('=', 1)
        res = read_json(Path(path) / 'results.json')
        runs[label] = {r['key']: (r['drift_pct'] if r['status'] == 'scored' else np.nan) for r in res}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())) - set((args.exclude or '').split(',')))
    groups = [('all', keys), ('all except zoom', [k for k in keys if mode[k] != 'zoom'])]
    groups += [(s, [k for k in keys if split[k] == s]) for s in ('train', 'val')]
    groups += [(m, [k for k in keys if mode[k] == m])
               for m in sorted({mode[k] for k in keys}, key=lambda m: -sum(mode[k] == m for k in keys))]
    labels = list(runs)
    base = labels[0]
    print(f"{'group':<17}{'n':>4}" + ''.join(f'{l:>22}' for l in labels))
    rows = []
    for name, ks in groups:
        if not ks:
            continue
        cells, row = '', dict(group=name, n=len(ks))
        for l in labels:
            v = np.array([runs[l][k] for k in ks], float)
            mean, med = np.nanmean(v), np.nanmedian(v)
            d = mean - np.nanmean([runs[base][k] for k in ks])
            txt = f'{mean:.2f}/{med:.2f}' + ('' if l == base else f' ({d:+.2f})')
            cells += f'{txt:>22}'
            row.update({f'{l}_mean': round(float(mean), 3), f'{l}_median': round(float(med), 3),
                        f'{l}_scored': int(np.isfinite(v).sum())})
        rows.append(row)
        print(f'{name:<17}{len(ks):>4}' + cells)
    print('cells: mean/median drift %, (change of the mean vs the first run)')
    if args.paired:
        print(f"\nper clip vs {base}: better / worse by > 0.5 points, mean change; [base < 10 %: better / worse]")
        for name, ks in groups:
            if not ks:
                continue
            b = np.array([runs[base][k] for k in ks], float)
            line = f'{name:<17}{len(ks):>4}'
            for l in labels[1:]:
                dl = np.array([runs[l][k] for k in ks], float) - b
                good = b < 10
                line += (f'  {l}: {(dl < -.5).sum()}/{(dl > .5).sum()} {np.nanmean(dl):+.2f}'
                         f' [{(dl[good] < -.5).sum()}/{(dl[good] > .5).sum()}]')
            print(line)
    if args.clips_of:
        wanted = set(args.clips_of.split(','))
        print(f"\n{'clip':<10}{'mode':<10}" + ''.join(f'{l:>14}' for l in labels))
        for k in keys:
            if mode[k] in wanted:
                print(f'{k:<10}{mode[k]:<10}' + ''.join(f'{runs[l][k]:>14.2f}' for l in labels))
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(f for r in rows for f in r)))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
