#!/usr/bin/env python3
"""Speed fixes: speed dirs for one fitted candidate (oracle.py --speedfix-tune, speedfix_best.json): the flow
variant, merged with its horizon-sanity variant by the fitted rule (common.hs_merge), for dev and test, plus the
filter params (v3a + near-band options) and where the rule fires (fraction of pairs replaced per clip). GT-free.

  speedfix_build.py --best runs/speedfix/tune-w32mean/speedfix_best.json --family hs --output runs/speedfix/speed-w32mean-hs
writes OUTPUT-dev/speed, OUTPUT-test/speed, OUTPUT.params.json, OUTPUT.fires.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NO_LIMIT, hs_merge, write_json  # noqa: E402

V3A = Path('runs/speed-tune3/params_v3a_no_hjump.json')
PREFIX = {'dev': 'runs/flow-sf', 'test': 'runs/flow-sf-test'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--best', required=True, type=Path)
    ap.add_argument('--family', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    b = json.loads(args.best.read_text())[args.family]
    params = json.loads(V3A.read_text())
    if b.get('near_w') not in (None, ''):
        params.update(near_w=float(b['near_w']), near_reliable=b['near_reliable'] in (True, 'True'))
    write_json(Path(f'{args.output}.params.json'), params)
    fires = {}
    for split, prefix in PREFIX.items():
        out = Path(f'{args.output}-{split}') / 'speed'
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(f"{prefix}-{b['flow']}/speed").glob('*.npz')):
            d = dict(np.load(f))
            if b.get('hs') not in (None, ''):
                h = Path(f"{prefix}-{b['hs']}/speed") / f.name
                P = NO_LIMIT if b['P'] in (None, '') else float(b['P'])
                C = NO_LIMIT if b['C'] in (None, '') else float(b['C'])
                d, frac = hs_merge(d, dict(np.load(h)), P, C)
                fires[f.stem] = dict(split=split, replaced=frac)
            np.savez(out / f.name, **d)
    write_json(Path(f'{args.output}.fires.json'), fires)
    print(args.family, b['label'], 'clips where the rule fires:',
          {s: [k for k, v in fires.items() if v['split'] == s and v['replaced'] > 0] for s in PREFIX})


if __name__ == '__main__':
    main()
