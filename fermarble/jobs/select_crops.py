#!/usr/bin/env python3
"""Own-bow crops from the static-region detector (static_regions.py -> static_regions.csv): CPU, no references.

Every clip whose bow_crop_frac lies in [CROP_MIN, CROP_MAX] gets that crop, whatever its camera type; no clip is
selected or excluded by ID. Writes {clip: fraction of the frame height to drop at the bottom}, rounded to 3 decimals,
for the --crops option of pi3_tool.py / omega_tool.py (and --crop-bottom of run_dpvo.py).

  select_crops.py --output runs/static-regions/crops_v2.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_regions import BLOCK_MIN, MAX_CROP  # noqa: E402

CROP_MIN = BLOCK_MIN  # 3 %: the detector's own threshold for counting a bottom block as a bow block
CROP_MAX = 0.35       # values near the detector's 50 % cap (MAX_CROP) are treated as false detections
assert CROP_MAX < MAX_CROP


def select(rows):
    return {r['key']: round(float(r['bow_crop_frac']), 3) for r in rows
            if CROP_MIN <= float(r['bow_crop_frac']) <= CROP_MAX}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', type=Path, default=Path('runs/static-regions/static_regions.csv'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.output.exists() and not args.overwrite:
        ap.error(f'{args.output} exists (use --overwrite)')
    rows = list(csv.DictReader(open(args.csv)))
    sel = select(rows)
    args.output.write_text(json.dumps(sel, indent=1))
    split = {r['key']: r['split'] for r in rows}
    print('written', args.output, f'({len(sel)} clips:',
          ', '.join(f"{sum(split[k] == s for k in sel)} {s}" for s in ('train', 'val', 'test')) + ')')


if __name__ == '__main__':
    main()
