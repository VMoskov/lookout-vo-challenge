#!/usr/bin/env python3
"""Static image regions per clip (own bow, mount, droplets, HUD): CPU, no references.

Per clip: N evenly spaced frames, grey, active rect only (manifest calibration), downscaled to
STATIC_W px wide. static_frac: per-pixel temporal std < --std-max grey levels (exposure-sensitive;
misses bows under changing exposure, kept for reference). Rigid structure: Canny edges that are
present at the same pixel in >= EDGE_PERSIST of the frames (water edges are random; the horizon,
logos and HUD also persist). Bottom block (own bow, deck, mount, radar dome): components (dilated) of
edges persistent in >= RIGID_PERSIST of the frames that reach the bottom rows (tall thin masts/poles are
reported but not cropped; components in the outer 15 % of the width (HUD) or narrower than 5 % are ignored), area >= --min-area of the frame. bow_crop_frac =
height from the bottom to the top of the bottom block, as a fraction of the active height (<= MAX_CROP). Writes CSV + per-clip mask
PNGs + a contact sheet of the clips with a bottom block.
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

STATIC_W = 320
EDGE_PERSIST = 0.6    # edge present at a pixel in >= this fraction of frames = rigid structure
MAX_CROP = 0.5
RIGID_PERSIST = 0.85  # bottom components: edges this persistent (wake/glint speckle stays below)
BORDER = 3
BLOCK_MIN = 0.03      # a bottom block counts as a bow block from 3 % of the height
MANIFESTS = [Path('release/release_manifest.json'), Path('lookout-vo-test-v0.1/release_manifest.json')]


def clip_static(args):
    root, c, n, std_max, min_area, row_frac, out = args
    cap = cv2.VideoCapture(str(root / c['video']))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, y0, x1, y1 = c['calibration']['active_rect']
    frames, edges = [], []
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
    F = np.stack(frames).astype(np.float32)
    static = (F.std(0) < std_max).astype(np.uint8)                  # raw intensity variance (exposure-sensitive)
    P = np.mean(edges, 0)
    P[:, :BORDER] = P[:, -BORDER:] = 0                                # frame-border artefacts
    persist = (P >= EDGE_PERSIST).astype(np.uint8)                    # edge at the same pixel in most frames
    rigid = (P >= RIGID_PERSIST).astype(np.uint8)                     # own-boat structure: edge in nearly every frame
    H, W = persist.shape
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(rigid, np.ones((5, 5), np.uint8)), connectivity=8)
    bottom = np.zeros_like(persist)
    masts = []
    touch = set(lab[H - max(1, H // 30):].ravel()) - {0}             # components reaching the bottom rows
    for i in touch:
        if stats[i, cv2.CC_STAT_AREA] < min_area * persist.size:
            continue
        if stats[i, cv2.CC_STAT_HEIGHT] > 0.6 * H and stats[i, cv2.CC_STAT_WIDTH] < 0.15 * W:
            masts.append(i)                                           # mast/pole: a bottom crop cannot remove it
            continue
        xl, ww = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        if xl + ww <= 0.15 * W or xl >= 0.85 * W or ww < 0.05 * W:
            continue                                                  # corner HUD (gauge, icons) or speck
        bottom |= (lab == i).astype(np.uint8)
    ys = np.where(bottom.any(1))[0]
    crop = min(MAX_CROP, (H - ys.min()) / H) if len(ys) else 0.0
    out.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(np.median(F, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[persist > 0] = (0, 0, 200)
    vis[(bottom > 0) & (rigid > 0)] = (0, 220, 0)
    for i in masts:
        vis[(lab == i) & (rigid > 0)] = (255, 120, 0)
    if crop:
        cv2.line(vis, (0, int(H * (1 - crop))), (W - 1, int(H * (1 - crop))), (0, 255, 255), 1)
    cv2.imwrite(str(out / f"{c['key']}.png"), vis)
    t30 = int(round(0.3 * H))
    return dict(key=c['key'], split=c['split'], camera_mode=c['camera_mode'], n_frames=len(frames),
                static_frac=float(static.mean()), persist_edge_frac=float(persist.mean()),
                persist_bottom30=float(persist[H - t30:].mean()), bottom_block_frac=float((bottom & rigid).mean()),
                mast=bool(masts),
                bow_crop_frac=float(crop), median_std=float(np.median(F.std(0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--std-max', type=float, default=2.0)
    ap.add_argument('--min-area', type=float, default=0.002)
    ap.add_argument('--row-frac', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('runs/static-regions'))
    args = ap.parse_args()
    jobs = []
    for m in MANIFESTS:
        man = json.loads(m.read_text())
        jobs += [(m.parent, c, args.frames, args.std_max, args.min_area, args.row_frac, args.output / 'masks')
                 for c in man['clips']]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(clip_static, jobs))
    with open(args.output / 'static_regions.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hit = [r for r in rows if r['bow_crop_frac'] >= BLOCK_MIN]
    if hit:
        tiles = [cv2.imread(str(args.output / 'masks' / f"{r['key']}.png")) for r in hit]
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        for t, r in zip(tiles, hit):
            cv2.putText(t, f"{r['key'][5:]} {r['camera_mode']} crop {100 * r['bow_crop_frac']:.0f}%", (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cols = 6
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(args.output / 'bottom_blocks.jpg'), sheet)
    print('written', args.output / 'static_regions.csv', f'({len(hit)} clips with a bottom block >= {BLOCK_MIN:.0%})')


if __name__ == '__main__':
    main()
