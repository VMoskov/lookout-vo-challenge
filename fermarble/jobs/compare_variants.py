#!/usr/bin/env python3
"""Compare dev prediction runs (each scored by starter/evaluate.py) against a base run.

Table per variant: val / val wide / train / all dev / per camera type, mean/median drift %,
clips better/worse than the base (> 0.05 points), worst single loss; plus how many test clips
change vs the base's test predictions. Writes a markdown section and a JSON verdict
(beats_both = lower mean than the base on both train and val). Reads only evaluator output.

  compare_variants.py --base LABEL=DEVDIR[:TESTDIR] --variant LABEL=DEVDIR[:TESTDIR] ... --output STEM
"""
import argparse
import filecmp
import json
from pathlib import Path

import numpy as np


def spec(s):
    label, rest = s.split('=', 1)
    dev, _, test = rest.partition(':')
    return label, Path(dev), Path(test) if test else None


def scores(d):
    return {r['key']: r['drift_pct'] for r in json.loads((d / 'results.json').read_text())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, type=spec)
    ap.add_argument('--variant', required=True, nargs='+', type=spec)
    ap.add_argument('--title', default='variants')
    ap.add_argument('--output', required=True, type=Path, help='output stem: writes STEM.md and STEM.json')
    args = ap.parse_args()
    bl, bdev, btest = args.base
    B = scores(bdev)
    m = json.loads((bdev / 'prediction_manifest.json').read_text())['clips']
    modes = sorted({v['camera_mode'] for v in m.values()})
    groups = [('val', lambda k: m[k]['split'] == 'val'),
              ('val wide', lambda k: m[k]['split'] == 'val' and m[k]['camera_mode'] == 'wide'),
              ('train', lambda k: m[k]['split'] == 'train'), ('all dev', lambda k: True)]
    groups += [(md, lambda k, md=md: m[k]['camera_mode'] == md) for md in modes]
    L = [f'### {args.title}', '', f'Cells: mean / median (clips better+ / worse- vs {bl}, worst single loss).', '',
         '| run | ' + ' | '.join(g for g, _ in groups) + ' | test clips changed | beats base on train and val |',
         '|---' * (len(groups) + 3) + '|']
    verdict = {}
    for label, dev, test in [args.base] + args.variant:
        try:
            X = scores(dev)
        except FileNotFoundError:
            L.append(f'| {label} | missing |')
            continue
        cells, means = [], {}
        for g, f in groups:
            ks = [k for k in B if f(k)]
            v = np.array([X[k] for k in ks])
            dd = v - np.array([B[k] for k in ks])
            means[g] = float(v.mean())
            cells.append(f'{v.mean():.2f} / {np.median(v):.2f} ({(dd < -0.05).sum()}+ / {(dd > 0.05).sum()}- , '
                         f'worst {dd.max():+.1f})')
        changed = None
        if test is not None and btest is not None and (test / 'predictions').is_dir():
            changed = [p.stem for p in sorted((btest / 'predictions').glob('*.json'))
                       if not filecmp.cmp(p, test / 'predictions' / p.name, shallow=False)]
        beats = label != bl and means['train'] < np.mean([B[k] for k in B if m[k]['split'] == 'train']) \
            and means['val'] < np.mean([B[k] for k in B if m[k]['split'] == 'val'])
        verdict[label] = dict(means=means, test_changed=changed, beats_both=bool(beats))
        L.append(f'| {label} | ' + ' | '.join(cells) + f" | {'-' if changed is None else len(changed)} | "
                 f"{'-' if label == bl else ('**yes**' if beats else 'no')} |")
    for label, v in verdict.items():
        if v['test_changed']:
            L.append(f"\n- {label}: test clips changed vs {bl}: {', '.join(c[5:] for c in v['test_changed'])}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.md').write_text('\n'.join(L) + '\n')
    args.output.with_suffix('.json').write_text(json.dumps(verdict, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
