#!/usr/bin/env python3
"""Visual check of the horizon sanity rule (the fitted 'hs' family of runs/speedfix/tune-w32mean) on a few clips.

Per clip, ~4 frames (spread over the clip, at least half from windows where the rule fires when it fires anywhere),
one row each: frame | BEFORE | AFTER. BEFORE: PanSR overlay (water blue, sky grey, other classes = own boat /
obstacles / shore orange), the PanSR water edge (white), DPVO's mount-corrected horizon (red), the segmentation
horizon shifted up by U px (cyan), and the old near / far band sample points (green / yellow). AFTER: the same with
the new band points (magenta / orange): the old ones where the rule does not fire in that window. Bands are
recomputed exactly as flow_speed.py does ('none' variant: water_region below the horizon, minus never-moving
points; split at the median depression), with the static mask from this clip's flow.
Below: old vs new speed (as submitted, relative), the unfiltered reliable-pair profiles (what the speed would be
without the clip / gate fallback), and the windows where the rule fires shaded.
Writes runs/speedfix-vis/<clip>.png and contact_sheet.png. Reads model outputs only (no references).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (filter_flow_speed, flow_reliable_speed, flow_speed_from_reliable, hs_merge, load_manifest,  # noqa: E402
                    load_timestamps, select_clips)
from flow_speed import (GRID_STEP, SEG_SKY, SEG_WATER, STATIC_FLOW_PX, clip_frames, horizon_rows,  # noqa: E402
                        seg_horizon_ups, water_region)

O = Path('runs/speedfix-vis')
SF = Path('runs/speedfix')
SETS = {'test': (Path('lookout-vo-test-v0.1/release_manifest.json'), 'runs/flow-sf-test', 'test'),
        'dev': (Path('release/release_manifest.json'), 'runs/flow-sf', 'dev')}
BGR = dict(red=(0, 0, 255), cyan=(255, 255, 0), white=(255, 255, 255), old_near=(0, 220, 0), old_far=(0, 230, 255),
           new_near=(255, 0, 255), new_far=(0, 140, 255))


def overlay(gray, seg, j, H, W):
    """Frame with the PanSR classes of segmented frame j blended in."""
    m = cv2.resize(seg['masks'][j], (W, H), interpolation=cv2.INTER_NEAREST)
    col = np.zeros((H, W, 3), np.uint8)
    col[m == SEG_WATER] = (200, 90, 0)
    col[m == SEG_SKY] = (150, 150, 150)
    col[(m != SEG_WATER) & (m != SEG_SKY)] = (0, 120, 255)
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(im, 0.6, col, 0.4, 0)


def polyline(im, u, K, color, W, H, thick=2):
    xs = np.arange(0, W, 4.0)
    y = horizon_rows(u, K, xs)
    ok = np.isfinite(y) & (y >= 0) & (y < H)
    if ok.sum() > 1:
        cv2.polylines(im, [np.column_stack([xs[ok], y[ok]]).astype(np.int32)], False, color, thick)


def bands(u, rays, static):
    water, s = water_region(rays, u / np.linalg.norm(u), static)
    if not water.any():
        return water, water
    split = np.median(s[water])
    return water & (s >= split), water & (s < split)


def render_clip(split_set, key, best, params):
    manifest, flow_prefix, sp = SETS[split_set]
    clip = {c['key']: c for c in select_clips(load_manifest(manifest), 'test' if sp == 'test' else 'train,val', None)}[key]
    root = manifest.parent
    U, P, C = float(best['hs'][2:]), best['P'], best['C']
    d_old = dict(np.load(f'{flow_prefix}-none/speed/{key}.npz'))
    d_hs = dict(np.load(f"{flow_prefix}-{best['hs']}/speed/{key}.npz"))
    d_new, frac = hs_merge(d_old, d_hs, P, C)
    seg = dict(np.load(f'runs/seg-pansr/{key}.npz'))
    frames, dec, times, K = clip_frames(root, clip, 8.0, float(d_old['focal_scale']))
    assert np.array_equal(dec, d_old['frame_indices'])
    H, W = frames[0].shape
    gy, gx = np.mgrid[GRID_STEP // 2:H:GRID_STEP, GRID_STEP // 2:W:GRID_STEP]
    p = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    fx, fy, cx, cy = K
    rays = np.column_stack([(p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, np.ones(len(p))])
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    mag = np.empty((len(frames) - 1, len(p)), np.float32)
    for k in range(len(frames) - 1):
        mag[k] = np.linalg.norm(dis.calc(frames[k], frames[k + 1], None)[gy, gx].reshape(-1, 2), axis=1)
    static = np.percentile(mag, 80, axis=0) < STATIC_FLOW_PX
    del mag
    u_dpvo = d_old['dpvo_up']
    u_seg = seg_horizon_ups(seg, times, K, W, H, shift_px=U)[0]
    fire = (d_hs['hs_window_below'] > P) | (d_hs['hs_window_cross'] > C)
    t0s, win = d_hs['hs_window_t0'], np.clip(((times - times[0]) // 20.0).astype(int), 0, len(d_hs['hs_window_t0']) - 1)
    # pairs hs_merge takes from the hs variant (fired windows are a subset of its candidate windows: P, C > the
    # candidate thresholds) whose segmentation horizon exists
    replaced = np.r_[fire[d_hs['hs_pair_window']] & d_hs['hs_cand'], False] & np.isfinite(u_seg).all(1)

    # frames: centres of fired windows and of the others, spread over the clip (only segmented frames)
    sf = seg['seg_frames']
    cen = lambda ws: [int(sf[np.argmin(np.abs(times[sf] - (t0s[w] + 10.0)))]) for w in ws]
    pick = lambda ws, n: [ws[i] for i in np.unique(np.linspace(0, len(ws) - 1, n).round().astype(int))] if ws else []
    fw, nw = [int(w) for w in np.nonzero(fire)[0]], [int(w) for w in np.nonzero(~fire)[0]]
    n_f = 4 if not nw else 0 if not fw else 2
    ks = sorted(cen(pick(fw, n_f)) + cen(pick(nw, 4 - n_f)))

    rows = []
    for k in ks:
        j = int(np.nonzero(sf == k)[0][0])
        kk = min(k, len(d_old['speed_near']) - 1)
        base = overlay(frames[k], seg, j, H, W)
        top = seg['water_top'][j]
        okt = (top > 0) & (top < H - 1)
        for x in np.nonzero(okt)[0][::3]:
            cv2.circle(base, (int(x), int(top[x])), 1, BGR['white'], -1)
        polyline(base, u_dpvo[k], K, BGR['red'], W, H)
        if u_seg is not None and np.isfinite(u_seg[k]).all():
            polyline(base, u_seg[k], K, BGR['cyan'], W, H)
        on = bool(replaced[k])
        u_new = u_seg[k] if on else u_dpvo[k]
        panels = []
        for tag, u, d, cn, cf in (('BEFORE', u_dpvo[k], d_old, 'old_near', 'old_far'),
                                  ('AFTER', u_new, d_new, 'new_near', 'new_far')):
            im = base.copy()
            nr, fr = bands(u, rays, static)
            for sel, c in ((nr, cn), (fr, cf)):
                for x, y in p[sel].astype(int):
                    cv2.circle(im, (x, y), 2, BGR[c], -1)
            sn, sfar = d['speed_near'][kk], d['speed_far'][kk]
            ratio = f'near/far {sn / sfar:.2f}' if np.isfinite(sn) and np.isfinite(sfar) and sfar > 0 else 'near/far n/a'
            txt = f"{tag}{' (rule fires)' if tag == 'AFTER' and on else ' (rule off: = before)' if tag == 'AFTER' else ''}: " \
                  f"{nr.sum()}/{fr.sum()} pts, {ratio}"
            cv2.putText(im, txt, (8, 26), 0, 0.7, (0, 0, 0), 4)
            cv2.putText(im, txt, (8, 26), 0, 0.7, (255, 255, 255), 1)
            panels.append(im)
        raw = cv2.cvtColor(frames[k], cv2.COLOR_GRAY2BGR)
        w_ = int(win[k])
        lab = f"{key[5:]} t={times[k] - times[0]:.0f}s  window below {d_hs['hs_window_below'][w_]:.0f}px cross {d_hs['hs_window_cross'][w_]:.2f}"
        cv2.putText(raw, lab, (8, 26), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(raw, lab, (8, 26), 0, 0.7, (0, 255, 255), 1)
        rows.append(np.hstack([raw, panels[0], panels[1]]))
    grid = np.vstack(rows)

    # speed plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ft = load_timestamps(root, clip)
    head = lambda v: np.load(SF / f'{split_set}-w32mean-{v}/heading/{key}.npz')
    s_old, s_new = head('base')['speed'], head('hs')['speed']
    fig, ax = plt.subplots(figsize=(grid.shape[1] / 100, 3.2), dpi=100)
    for w in np.nonzero(fire)[0]:
        ax.axvspan(t0s[w] - ft[0], min(t0s[w] + 20, ft[-1]) - ft[0], color='tab:red', alpha=0.12, lw=0)
    info = {}
    for tag, d, col in (('old', d_old, 'tab:green'), ('new', d_new, 'tab:purple')):
        t, v, rel, _ = flow_reliable_speed(d, params['agree_tol'], params['side_max'], params['drop_backward'])
        _, inf = filter_flow_speed(d, ft, params)
        info[tag] = inf
        ok = np.isfinite(v)
        ax.plot(t[ok] - ft[0], v[ok], '.', ms=1.5, color=col, alpha=0.35)
        if ok.sum() > 10:
            prof, _ = flow_speed_from_reliable(t, v, ft, params['window_s'], params['clamp_lo'], params['clamp_hi'])
            ax.plot(ft - ft[0], prof, ':', color=col, lw=1.2,
                    label=f'{tag} flow profile before fallback (reliable {rel:.0%})')
    ax.plot(ft - ft[0], s_old, color='tab:green', lw=2, label=f"old speed as submitted ({info['old']['speed_fallback']})")
    ax.plot(ft - ft[0], s_new, '--', color='tab:purple', lw=2, label=f"new speed as submitted ({info['new']['speed_fallback']})")
    for k in ks:
        ax.axvline(times[k] - ft[0], color='k', lw=0.6, alpha=0.5)
    cv_ = lambda i: f"cv {i['flow_cv']:.2f}" if 'flow_cv' in i else ''
    ax.set_title(f"{key} {clip['camera_mode']} ({sp}): relative speed; red = windows where the rule fires (U{U:g} P{P:g} C{C:g}), "
                 f"{frac:.0%} of pairs replaced; old {cv_(info['old'])} new {cv_(info['new'])}; gate {params['gate_min']}; "
                 f"black lines = rendered frames", fontsize=9)
    ax.set_ylim(0, min(5.0, max(2.5, 1.1 * max(s_old.max(), s_new.max()))))
    ax.set_xlim(0, ft[-1] - ft[0])
    ax.set_xlabel('s')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.canvas.draw()
    plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    plot = cv2.resize(plot, (grid.shape[1], int(plot.shape[0] * grid.shape[1] / plot.shape[1])))
    head_ = np.full((40, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head_, f"{key} {clip['camera_mode']}: frame | BEFORE (DPVO horizon red, old bands near green / far yellow) | "
                       f"AFTER (new bands near magenta / far orange); cyan = segmentation horizon +{U:g}px up, white = PanSR water edge; "
                       f"overlay water blue, sky grey, other orange", (8, 27), 0, 0.6, (0, 0, 0), 1)
    out = np.vstack([head_, grid, plot])
    cv2.imwrite(str(O / f'{key}.png'), out)
    return key, out, dict(fires_windows=[round(float(t0s[w] - times[0])) for w in np.nonzero(fire)[0]],
                          replaced=frac, frames_s=[round(float(times[k] - times[0]), 1) for k in ks],
                          old=info['old']['speed_fallback'], new=info['new']['speed_fallback'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', nargs='+', default=['test:clip_103', 'test:clip_104', 'dev:clip_050'])
    args = ap.parse_args()
    O.mkdir(parents=True, exist_ok=True)
    best = json.loads((SF / 'tune-w32mean/speedfix_best.json').read_text())['hs']
    params = json.loads((SF / 'speed-w32mean-hs.params.json').read_text())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(len(args.clips)) as ex:
        res = list(ex.map(render_clip, *zip(*[(c.split(':')[0], c.split(':')[1], best, params) for c in args.clips])))
    width = 1600
    tiles = [cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))) for _, im, _ in res]
    cv2.imwrite(str(O / 'contact_sheet.png'), np.hstack([cv2.copyMakeBorder(t, 0, max(x.shape[0] for x in tiles) - t.shape[0], 0, 10,
                                                                             cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]))
    for k, _, s in res:
        print(k, s)


if __name__ == '__main__':
    main()
