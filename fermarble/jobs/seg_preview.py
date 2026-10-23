#!/usr/bin/env python3
"""Segmentation preview, stage B (PanSR env, GPU): PanSR (LaRS-pretrained, external pretraining,
github.com/lojzezust/PanSR, Apache-2.0) on the frames from seg_preview_frames.py. Preview only:
nothing in the pipeline uses it.

Per frame: frame | overlay of the four groups (water, sky, static obstacle, dynamic obstacles =
LaRS thing classes) with a legend, the flow_speed.py horizon (red) and its water sample points
(near band green, far band yellow). Writes <output>/<clip>_<i>.png, contact_sheet.jpg and
seg_preview.csv (group fractions; share of flow-band points that PanSR labels water).
Run with ~/micromamba/envs/pansr/bin/python from lookout-vo-challenge.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

# LaRS categories (dataset id = contiguous id): 1-8 things, 9 static obstacle, 10 water, 11 sky
LARS = ['Boat/ship', 'Row boat', 'Paddle board', 'Buoy', 'Swimmer', 'Animal', 'Float', 'Other',
        'Static Obstacle', 'Water', 'Sky']
GROUPS = {'water': ([10], (200, 120, 30)), 'sky': ([11], (235, 206, 135)),
          'static obstacle': ([9], (40, 140, 230)), 'dynamic obstacles': (list(range(1, 9)), (180, 40, 220))}


def register_metadata(name):
    """The LaRS metadata PanSR needs at build time, without the LaRS annotation files."""
    from detectron2.data import MetadataCatalog
    things = list(range(1, 9))
    MetadataCatalog.get(name).set(
        thing_classes=['void'] + LARS[:8], stuff_classes=['void'] + LARS,
        thing_colors=[[0, 0, 0]] * 9, stuff_colors=[[0, 0, 0]] * 12,
        thing_dataset_id_to_contiguous_id={i: i for i in things},
        stuff_dataset_id_to_contiguous_id={i: i for i in range(1, 12)},
        ignore_label=255, label_divisor=256 * 256)


def load_predictor(repo):
    import pansr  # noqa: F401  (registers the architecture)
    from detectron2.engine.defaults import DefaultPredictor
    from pansr.hub import PanSRHF
    for name in ('lars_train_panoptic', 'lars_val_panoptic'):   # the model reads the train name
        register_metadata(name)
    hf = PanSRHF.from_pretrained(repo)
    predictor = DefaultPredictor(hf.cfg)
    predictor.model = hf.model.to(hf.cfg.MODEL.DEVICE).eval()
    return predictor


def class_map(predictor, img):
    with torch.no_grad():
        pan, info = predictor(img)['panoptic_seg']
    pan = pan.cpu().numpy()
    cls = np.zeros(pan.shape, np.int32)
    for s in info:
        cls[pan == s['id']] = s['category_id']
    return cls


def render(d, cls, width=640):
    img = d['image']
    H, W = img.shape[:2]
    over = img.copy()
    for name, (ids, col) in GROUPS.items():
        m = np.isin(cls, ids)
        over[m] = (0.45 * over[m] + 0.55 * np.array(col)).astype(np.uint8)
    if d['has_band']:
        for pts, col in ((d['far'], (0, 255, 255)), (d['near'], (0, 255, 0))):
            for x, y in pts.astype(int):
                cv2.circle(over, (x, y), 2, col, -1)
        hz = d['horizon']
        hz = hz[np.isfinite(hz[:, 1]) & (hz[:, 1] >= 0) & (hz[:, 1] < H)].astype(np.int32)
        if len(hz) > 1:
            cv2.polylines(over, [hz.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    pair = np.hstack([img, over])
    s = width * 2 / pair.shape[1]
    pair = cv2.resize(pair, (width * 2, int(pair.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bar = np.full((44, pair.shape[1], 3), 30, np.uint8)
    title = f"{d['key']} {d['camera_mode']} ({d['split']}) t={float(d['time']):.0f}s"
    cv2.putText(bar, title, (6, 16), 0, 0.5, (255, 255, 255), 1)
    x = 6
    items = [(n, c) for n, (_, c) in GROUPS.items()] + [('flow near', (0, 255, 0)), ('flow far', (0, 255, 255)),
                                                         ('horizon', (0, 0, 255))]
    if not d['has_band']:
        items = items[:4] + [('no flow band (no flow file)', (128, 128, 128))]
    for n, c in items:
        cv2.rectangle(bar, (x, 26), (x + 12, 38), c, -1)
        cv2.putText(bar, n, (x + 16, 37), 0, 0.42, (255, 255, 255), 1)
        x += 22 + 8 * len(n)
    return np.vstack([bar, pair])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=Path, default=Path('runs/seg-preview/frames'))
    ap.add_argument('--output', type=Path, default=Path('runs/seg-preview'))
    ap.add_argument('--model', default='lojzezust/pansr-lars-swinl')
    args = ap.parse_args()
    predictor = load_predictor(args.model)
    rows, tiles = [], []
    for f in sorted(args.frames.glob('*.npz')):
        d = dict(np.load(f, allow_pickle=False))
        d = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}
        cls = class_map(predictor, d['image'])
        tile = render(d, cls)
        cv2.imwrite(str(args.output / f'{f.stem}.png'), tile)
        tiles.append((f.stem, tile))
        row = dict(frame=f.stem, key=d['key'], split=d['split'], camera_mode=d['camera_mode'], time=round(float(d['time']), 1))
        row.update({g: round(float(np.isin(cls, ids).mean()), 3) for g, (ids, _) in GROUPS.items()})
        if d['has_band']:
            pts = np.concatenate([d['near'], d['far']]).astype(int)
            on = cls[pts[:, 1], pts[:, 0]] if len(pts) else np.array([])
            row['band_points'] = len(pts)
            row['band_on_water'] = round(float(np.isin(on, [10]).mean()), 3) if len(pts) else ''
            row['band_on_static'] = round(float(np.isin(on, [9]).mean()), 3) if len(pts) else ''
            row['band_on_dynamic'] = round(float(np.isin(on, list(range(1, 9))).mean()), 3) if len(pts) else ''
        rows.append(row)
        print(f.stem, {k: row[k] for k in row if k not in ('frame', 'split', 'time')}, flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(args.output / 'seg_preview.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # contact sheet: one row per clip (its 3 frames), tiles at half size
    by_clip = {}
    for stem, t in tiles:
        by_clip.setdefault(stem.rsplit('_', 1)[0], []).append(cv2.resize(t, (t.shape[1] // 2, t.shape[0] // 2)))
    width = max(sum(x.shape[1] for x in v) for v in by_clip.values())
    rows_img = []
    for v in by_clip.values():
        h = max(x.shape[0] for x in v)
        r = np.hstack([cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in v])
        rows_img.append(cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT))
    cv2.imwrite(str(args.output / 'contact_sheet.jpg'), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('written', args.output)


if __name__ == '__main__':
    main()
