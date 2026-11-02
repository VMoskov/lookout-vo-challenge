#!/usr/bin/env python3
"""Merged pose dirs (relative symlinks) picking, per clip, one of two runs by a rule on the clip, never by clip ID.

  --rule thermal: clips whose camera mode is in thermal.THERMAL_MODES take --alt (the CLAHE DPVO run), all others
                  --base (the plain DPVO run).
  --rule crops:   clips listed in --crops take --alt (the cropped run; its crop_bottom must equal the listed
                  fraction), all others --base (an uncropped run).

  merge_runs.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --rule thermal \
      --base runs/dpvo-test-fps8-seed0 --alt runs/dpvo-test-fps8-seed0-clahe --output runs/merged-test-seed0-clahe-v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thermal  # noqa: E402

MANIFEST = Path('release/release_manifest.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=MANIFEST)
    ap.add_argument('--split', default='test', help='train, val, test, comma-separated')
    ap.add_argument('--rule', choices=('thermal', 'crops'), required=True)
    ap.add_argument('--crops', type=Path, help='JSON {clip: fraction} (select_crops.py), for --rule crops')
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--alt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.rule == 'crops' and not args.crops:
        ap.error('--rule crops needs --crops')
    if args.output.exists():
        ap.error(f'{args.output} exists; merged dirs are never overwritten')
    crops = json.loads(args.crops.read_text()) if args.crops else {}
    clips = [c for c in json.loads(args.dataset.read_text())['clips'] if c['split'] in args.split.split(',')]
    picks = {}
    for c in clips:
        k = c['key']
        alt = c['camera_mode'] in thermal.THERMAL_MODES if args.rule == 'thermal' else k in crops
        src = (args.alt if alt else args.base) / 'poses' / f'{k}.npz'
        if not src.is_file():
            sys.exit(f'{k}: missing {src}')
        if args.rule == 'crops':
            want = float(crops.get(k, 0.0))
            z = np.load(src)
            got = float(z['crop_bottom']) if 'crop_bottom' in z.files else 0.0
            if got != want:
                sys.exit(f'{k}: {src} has crop_bottom {got}, crops file says {want}')
        picks[k] = src
    out = args.output / 'poses'
    out.mkdir(parents=True)
    for k, src in picks.items():
        (out / f'{k}.npz').symlink_to(os.path.relpath(src, out))
    n_alt = sum(p.parent.parent == args.alt for p in picks.values())
    print('written', args.output, f'({len(picks)} clips, {n_alt} from {args.alt}:',
          ', '.join(k for k, p in picks.items() if p.parent.parent == args.alt) + ')')


if __name__ == '__main__':
    main()
