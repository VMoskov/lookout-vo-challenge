#!/usr/bin/env python3
"""Inpainting masks for static own-boat structure that a bottom crop cannot remove (poles, masts,
a diagonal pole), and the bottom crop for a small own-boat object. CPU, no references.

Per clip: rigid edges (Canny edge at the same pixel in >= RIGID_PERSIST of 60 frames, as in
static_regions.py) on the active rect resized to static_regions.STATIC_W px wide; their
components (dilated 5x5) are kept if they are tall (> TALL of the height: poles, masts) or reach the
bottom rows (own-boat objects), and dropped if horizon-like (thin and wide) or in a corner (HUD).
With --crop-bottom-object, bottom objects are cropped instead (crop = height above their top) and
only the tall structures are inpainted. Mask = components closed (9x9), small holes (< 1 %) filled, dilated by
DILATE px. Writes <output>/<clip>.png (mask, active-rect coordinates at STATIC_W), <clip>_vis.png
and edits.json {clip: {"crop": fraction, "mask": path}} for vggt_tool.py --edits.

--auto: the same masks with NO per-clip arguments; every threshold lives in AUTO (below), fitted with
--fit on train+val clips only (never test) and frozen in <auto-dir>/config.json before any test clip
is processed. Rules, per clip (camera modes where Pi3X gives the heading):
  structure  = rigid components (horizon lines and corner HUD dropped as above) taller than `tall` of
               the height (poles, masts, a diagonal pole: a bottom crop cannot remove them)
  cropobj    = a bottom-touching component that is not tall and covers >= `cropobj_area` of the frame
               is cropped (crop = height above its top), not inpainted
  hull       = if the structure mask has >= 2 pieces within `hull_dist` px, each such group is
               replaced by its convex hull (parts of one instrument)
  selection  = inpaint only if the structure mask covers >= `area_min` of the frame; else no edit
  dilate     = one global value
Persistence maps are cached in <auto-dir>/cache.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STATIC_W = 320
RIGID_PERSIST = 0.85
TALL = 0.3
DILATE = 3
AUTO = dict(rigid_persist=RIGID_PERSIST, modes=['wide', 'flir-VIS'], tall=TALL, cropobj_area=0.002, hull_dist=0,
            area_min=0.01, dilate=DILATE)     # defaults; --fit replaces the fitted entries (config.json)
AUTO_GRID = dict(tall=[0.25, 0.3, 0.4], cropobj_area=[0.002, 0.005], hull_dist=[0, 5, 10, 20],
                 area_min=[0.005, 0.01, 0.02, 0.03], dilate=[2, 3, 4])
MANIFESTS = {'dev': Path('release/release_manifest.json'), 'test': Path('lookout-vo-test-v0.1/release_manifest.json')}


def persistence(root, c, n=60):
    cap = cv2.VideoCapture(str(root / c['video']))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = c['calibration']['active_rect']
    edges, frames = [], []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (STATIC_W, round(g.shape[0] * STATIC_W / g.shape[1])), interpolation=cv2.INTER_AREA)
        frames.append(g)
        edges.append(cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 100), np.ones((3, 3), np.uint8)) > 0)
    cap.release()
    P = np.mean(edges, 0)
    P[:, :3] = P[:, -3:] = 0
    return P, np.median(np.stack(frames), 0).astype(np.uint8)


def fill(mask, max_hole=0.01):
    """Close the components and fill their small holes (< max_hole of the frame); a large enclosed
    region (e.g. a rail and a pole closing a contour with the frame edge) is scene, not structure."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), connectivity=4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < max_hole * m.size:
            m[lab == i] = 1
    return m


def clip_mask(root, c, crop_bottom_object, hull=False, dilate=DILATE):
    P, med = persistence(root, c)
    rigid = (P >= RIGID_PERSIST).astype(np.uint8)
    H, W = rigid.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(rigid, np.ones((5, 5), np.uint8)), connectivity=8)
    tall, bottom = np.zeros_like(rigid), np.zeros_like(rigid)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 0.06 * H and w > 0.4 * W:
            continue                                   # horizon / shoreline line
        if x + w <= 0.15 * W and y + h <= 0.25 * H or x >= 0.85 * W and y + h <= 0.25 * H:
            continue                                   # corner HUD (logo)
        comp = (lab == i).astype(np.uint8)
        if h > TALL * H:
            tall |= comp
        elif y + h >= H - max(1, H // 30) and a >= 0.002 * H * W:
            bottom |= comp
    crop = 0.0
    if crop_bottom_object and bottom.any():
        crop = (H - np.where(bottom.any(1))[0].min()) / H
        bottom[:] = 0
    mask = fill(tall | bottom)
    if hull and mask.any():                            # moving parts of an instrument (wind vane): its convex hull
        n2, lab2, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n2):
            pts = cv2.findNonZero((lab2 == i).astype(np.uint8))
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return mask, crop, med


def cached_persistence(root, c, cache):
    p = cache / f"{c['key']}.npz"
    if p.is_file():
        z = np.load(p)
        return z['P'], z['med']
    P, med = persistence(root, c)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, P=P, med=med)
    return P, med


def auto_mask(P, cfg):
    """Mask, bottom-object crop and fired rules from a persistence map; cfg = AUTO-like dict."""
    rigid = (P >= cfg['rigid_persist']).astype(np.uint8)
    H, W = rigid.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(rigid, np.ones((5, 5), np.uint8)), connectivity=8)
    tall, crop, fired = np.zeros_like(rigid), 0.0, []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 0.06 * H and w > 0.4 * W:
            continue                                   # horizon / shoreline line
        if x + w <= 0.15 * W and y + h <= 0.25 * H or x >= 0.85 * W and y + h <= 0.25 * H:
            continue                                   # corner HUD (logo)
        if h > cfg['tall'] * H:
            tall |= (lab == i).astype(np.uint8)
        elif y + h >= H - max(1, H // 30) and a >= cfg['cropobj_area'] * H * W:
            crop = max(crop, (H - y) / H)              # cropobj
    mask = fill(tall)
    if cfg['hull_dist'] > 0 and mask.any():
        k = 2 * int(cfg['hull_dist']) + 1
        ng, glab = cv2.connectedComponents(cv2.dilate(mask, np.ones((k, k), np.uint8)), connectivity=8)
        for g in range(1, ng):
            piece = mask & (glab == g).astype(np.uint8)
            if cv2.connectedComponents(piece, connectivity=8)[0] - 1 >= 2:
                cv2.fillConvexPoly(mask, cv2.convexHull(cv2.findNonZero(piece)), 1)
                fired.append('hull')
    if mask.mean() < cfg['area_min']:
        return np.zeros_like(mask), 0.0, []
    fired = ['structure'] + sorted(set(fired)) + (['cropobj'] if crop else [])
    d = int(cfg['dilate'])
    return cv2.dilate(mask, np.ones((2 * d + 1, 2 * d + 1), np.uint8)), crop, fired


def iou_above(a, b, crop):
    """IoU of two masks above the crop line (the part that is inpainted; below it is cropped)."""
    r = int(round(a.shape[0] * (1 - crop)))
    a, b = a[:r] > 0, b[:r] > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def _persist_job(args):
    root, c, cache = args
    cached_persistence(root, c, cache)
    return c['key']


def run_auto(args):
    out = args.auto_dir
    cache = out / 'cache'
    split = args.split
    if split == 'test' and not (out / 'config.json').is_file():
        raise SystemExit('test needs a frozen config.json (run --fit on train+val first)')
    mp = MANIFESTS['test' if split == 'test' else 'dev']
    clips = [c for c in json.loads(mp.read_text())['clips']
             if (split == 'test' or c['split'] in ('train', 'val'))]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_persist_job, [(mp.parent, c, cache) for c in clips]))
    base = json.loads(args.base_crops.read_text()) if args.base_crops.is_file() else {}
    if args.fit:
        if split == 'test':
            raise SystemExit('--fit uses train+val only')
        import itertools
        target = set(args.target.split(','))
        hand = {k: cv2.imread(str(args.hand_dir / f'{k}.png'), cv2.IMREAD_GRAYSCALE) for k in target}
        maps = {c['key']: (c, cached_persistence(mp.parent, c, cache)[0]) for c in clips if c['camera_mode'] in AUTO['modes']}
        rows = []
        keys = list(AUTO_GRID)
        for combo in itertools.product(*(AUTO_GRID[k] for k in keys)):
            cfg = dict(AUTO, **dict(zip(keys, combo)))
            sel, ious = set(), []
            for k, (c, P) in maps.items():
                m, crop, _ = auto_mask(P, cfg)
                if m.any():
                    sel.add(k)
                    if k in hand:
                        ious.append(iou_above(m, hand[k], max(crop, float(base.get(k, 0.0)))))
            rows.append(dict(cfg=cfg, exact=sel == target, n_sel=len(sel), extra=sorted(sel - target),
                             missing=sorted(target - sel), iou=float(np.mean(ious)) if ious else 0.0))
        ok = [r for r in rows if r['exact']] or rows
        best = max(ok, key=lambda r: (r['exact'], r['iou']))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(dict(best['cfg'], fitted_on='train+val clips only', target=sorted(target),
                                                        dev_mean_iou_above_crop=round(best['iou'], 3),
                                                        n_exact_configs=sum(r['exact'] for r in rows), n_configs=len(rows)), indent=1))
        print(f"{sum(r['exact'] for r in rows)}/{len(rows)} configs select exactly {sorted(target)}; best: {best['cfg']} "
              f"mean IoU above crop {best['iou']:.3f}")
    cfg = {k: v for k, v in json.loads((out / 'config.json').read_text()).items() if k in AUTO}
    edits, report = {}, {}
    for c in clips:
        k = c['key']
        P, med = cached_persistence(mp.parent, c, cache)
        if c['camera_mode'] not in cfg['modes']:
            continue
        m, crop, fired = auto_mask(P, cfg)
        if not m.any():
            continue
        crop = max(crop, float(base.get(k, 0.0)))
        cv2.imwrite(str(out / f'{k}.png'), m * 255)
        vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
        vis[m > 0] = (0.4 * vis[m > 0] + [0, 0, 150]).astype(np.uint8)
        if crop:
            r = int(round(m.shape[0] * (1 - crop)))
            cv2.line(vis, (0, r), (m.shape[1] - 1, r), (0, 255, 255), 1)
        cv2.putText(vis, f"{k[5:]} {'+'.join(fired)}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(out / f'{k}_vis.png'), vis)
        edits[k] = dict(crop=round(crop, 3), mask=str(out / f'{k}.png'))
        hp = args.hand_dir / f'{k}.png'
        hand_iou = None
        if hp.is_file():
            hm = cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE)
            hand_iou = round(iou_above(m, hm, crop), 3) if hm.shape == m.shape else None
        report[k] = dict(split=c['split'], camera_mode=c['camera_mode'], rules=fired, mask_frac=round(float(m.mean()), 4),
                         crop=round(crop, 3), iou_vs_hand_above_crop=hand_iou)
        print(k, report[k])
    (out / f'edits_{split}.json').write_text(json.dumps(edits, indent=1))
    (out / f'report_{split}.json').write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--auto', action='store_true', help='automatic masks (AUTO rules, no per-clip arguments)')
    ap.add_argument('--fit', action='store_true', help='with --auto --split dev: fit AUTO_GRID on train+val, freeze config.json')
    ap.add_argument('--split', choices=['dev', 'test'], default='dev')
    ap.add_argument('--target', default='clip_005,clip_008,clip_012,clip_059', help='--fit: the dev clips to select')
    ap.add_argument('--hand-dir', type=Path, default=Path('runs/structure-masks'), help='hand masks (IoU reference)')
    ap.add_argument('--auto-dir', type=Path, default=Path('runs/structure-masks-auto'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--clips', help='SPLIT:clip_XXX[:cropobj][:hull][:dilate=N],... (cropobj = crop the '
                                                  'bottom object; hull = convex hull per component; dilate in px)')
    ap.add_argument('--base-crops', type=Path, default=Path('runs/static-regions/crops.json'),
                    help='existing bottom crops (kept; the larger crop wins)')
    ap.add_argument('--output', type=Path, default=Path('runs/structure-masks'))
    args = ap.parse_args()
    if args.auto:
        return run_auto(args)
    if not args.clips:
        ap.error('--clips is required without --auto')
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_crops.read_text()) if args.base_crops.is_file() else {}
    edits = {}
    for spec in args.clips.split(','):
        split, key, *opt = spec.split(':')
        mp = MANIFESTS[split]
        c = next(x for x in json.loads(mp.read_text())['clips'] if x['key'] == key)
        opts = dict(o.split('=') if '=' in o else (o, True) for o in opt)
        mask, crop, med = clip_mask(mp.parent, c, 'cropobj' in opts, 'hull' in opts, int(opts.get('dilate', DILATE)))
        crop = max(crop, float(base.get(key, 0.0)))
        cv2.imwrite(str(args.output / f'{key}.png'), mask * 255)
        vis = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
        vis[mask > 0] = (0.4 * vis[mask > 0] + [0, 0, 150]).astype(np.uint8)
        if crop:
            r = int(round(mask.shape[0] * (1 - crop)))
            cv2.line(vis, (0, r), (mask.shape[1] - 1, r), (0, 255, 255), 1)
        cv2.imwrite(str(args.output / f'{key}_vis.png'), vis)
        edits[key] = dict(crop=round(crop, 3), mask=str(args.output / f'{key}.png'))
        print(key, f'mask {mask.mean():.1%} of the frame, crop {crop:.1%}')
    (args.output / 'edits.json').write_text(json.dumps(edits, indent=1))


if __name__ == '__main__':
    main()
