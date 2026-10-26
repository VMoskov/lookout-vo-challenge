#!/usr/bin/env python3
"""Rotation-only bundle adjustment (exploratory, CPU, no references): refine Pi3X camera rotations
with relative rotations measured on static shore features, by rotation averaging.

Per clip (camera modes where Pi3X gives the heading):
1. The Pi3X keyframes (vggt_tool.keyframes: 2 fps, the same bottom crop and structure inpainting as
   the Pi3X run being refined) and their intrinsics K (518 px wide).
2. Usable pixels: PanSR static obstacle (seg_run.py, nearest segmented frame) minus everything that
   is static in the image (edge persistence >= STATIC_P, structure_masks cache: HUD, logos, own-boat
   structure), mapped through the same crop/resize as the keyframes.
3. SIFT matches (ratio test) at baselines BASELINES keyframes apart; relative rotation from the
   essential matrix (RANSAC, recoverPose). Kept if >= MIN_INLIERS inliers, inlier ratio >= 0.5 and
   within GROSS_DEG of Pi3X's relative rotation (gross-outlier check).
4. A clip is refined only with >= max(MIN_CONSTRAINTS, MIN_CONSTRAINT_FRAC x keyframes) constraints;
   otherwise its Pi3X poses are left unchanged (open sea, few static features).
5. Rotation averaging (scipy least_squares, sparse): unknowns are rotation corrections of every
   keyframe (the first fixed); residuals = consecutive Pi3X relative rotations (sigma_p) and the
   feature relative rotations (SIGMA_F, soft-L1). Camera centres are kept.
Writes <output>-sp<sigma_p>/poses/<clip>.npz in the Pi3X format (only refined clips) and
<output>-constraints/<clip>.npz + summary.json.
--k-sources: intrinsics for the relative rotations, one output prefix each (the matches are shared):
  nominal       the released K (keyframe geometry)                         -> <output>
  pi3x-clip     Pi3X's focal (median of its per-window FOV), centred pp    -> <output>-pi3x-clip
  pi3x-window   Pi3X's focal of the window containing each keyframe       -> <output>-pi3x-window
Points of a pair are normalised with each keyframe's own K, then the essential matrix is estimated
on normalised coordinates (threshold 1 px / mean focal).
"""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_manifest, load_pose_npz, quat_to_rotmat, rotmat_to_quat, select_clips  # noqa: E402

BASELINES = (1, 4, 10, 20)       # keyframes (0.5, 2, 5, 10 s at 2 fps)
MIN_MATCHES = 15
MIN_INLIERS = 12
GROSS_DEG = 5.0
MIN_CONSTRAINTS = 20
MIN_CONSTRAINT_FRAC = 0.1
SIGMA_F_DEG = 0.1
STATIC_P = 0.5
SIGMA_P_GRID = (0.05, 0.1, 0.3, 1.0)
STATIC_OBSTACLE = 9
MANIFESTS = {'dev': Path('release/release_manifest.json'), 'test': Path('lookout-vo-test-v0.1/release_manifest.json')}


def processed_size(clip):
    """(H, W) of the starter's preprocessed frame: half the active rect, trimmed to a multiple of 16."""
    x0, y0, x1, y1 = clip['calibration']['active_rect']
    h, w = (y1 - y0) // 2, (x1 - x0) // 2
    return h - h % 16, w - w % 16, h, w


def to_keyframe_geometry(mask, crop, width=518):
    """A processed-frame mask through the keyframe transform (bottom crop, resize, centre row crop)."""
    h = mask.shape[0]
    m = mask[:h - int(round(crop * h))] if crop else mask
    nh = round(m.shape[0] * width / m.shape[1])
    r = cv2.resize(m.astype(np.uint8), (width, nh), interpolation=cv2.INTER_NEAREST)
    ch = nh // 14 * 14
    o = (nh - ch) // 2
    return r[o:o + ch] > 0


def static_masks(clip, dec, crop, seg, persist):
    """Usable-pixel mask per keyframe (keyframe geometry)."""
    H, W, hh, ww = processed_size(clip)
    excl = cv2.resize((persist >= STATIC_P).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)[:H, :W]
    excl = cv2.dilate(excl, np.ones((9, 9), np.uint8)) > 0
    sf_dec = seg['frame_indices'][seg['seg_frames']]
    out = []
    for d in dec:
        j = int(np.argmin(np.abs(sf_dec - d)))
        m = cv2.resize(seg['masks'][j], (W, H), interpolation=cv2.INTER_NEAREST) == STATIC_OBSTACLE
        out.append(to_keyframe_geometry(m & ~excl, crop))
    return out


def rel_rot(Ra, Rb):
    """Relative rotation mapping camera-a coordinates to camera-b coordinates (camera-to-world inputs)."""
    return Rb.T @ Ra


def keyframe_Ks(z, K_nom, times, width=518):
    """Per-keyframe K for each source: nominal, Pi3X clip median focal, Pi3X per-window focal."""
    h = np.asarray(z['window_hfov_deg'], float)
    f = (width / 2) / np.tan(np.radians(h) / 2)
    cy = K_nom[1, 2]
    Kf = lambda fx: np.array([[fx, 0, width / 2], [0, fx, cy], [0, 0, 1.0]])
    wt = np.asarray(z['window_time'], float)
    near = np.abs(times[:, None] - wt[None, :]).argmin(1)
    return {'nominal': [K_nom] * len(times), 'pi3x-clip': [Kf(np.median(f))] * len(times),
            'pi3x-window': [Kf(f[i]) for i in near]}


def clip_constraints(args):
    split, clip, pi3x_dir, crop, mask_path, out_dirs = args
    out_dir = out_dirs['nominal'] if 'nominal' in out_dirs else next(iter(out_dirs.values()))
    import vggt_tool
    k = clip['key']
    try:
        mp = MANIFESTS[split]
        z = load_pose_npz(pi3x_dir / 'poses' / f'{k}.npz')
        frames, times, dec, _, _, K = vggt_tool.keyframes(mp.parent, clip, 2.0, 0.0, crop, mask_path)
        if len(dec) != len(z['frame_indices']) or not np.array_equal(dec, z['frame_indices']):
            return k, dict(error='keyframes differ from the Pi3X run')
        seg = dict(np.load(f'runs/seg-pansr/{k}.npz'))
        persist = np.load(f'runs/structure-masks-auto/cache/{k}.npz')['P']
        masks = static_masks(clip, dec, crop, seg, persist)
        R0 = quat_to_rotmat(z['poses'][:, 3:7])
        ok = np.isfinite(z['poses']).all(1)
        sift = cv2.SIFT_create(nfeatures=400)
        feats = [sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), m.astype(np.uint8) * 255) for f, m in zip(frames, masks)]
        bf = cv2.BFMatcher()
        Ks = keyframe_Ks(z, K, np.asarray(times, float))
        cons = {s: [] for s in out_dirs}
        for i in range(len(frames)):
            for b in BASELINES:
                j = i + b
                if j >= len(frames) or not (ok[i] and ok[j]):
                    continue
                (ki, di), (kj, dj) = feats[i], feats[j]
                if di is None or dj is None or len(ki) < MIN_MATCHES or len(kj) < MIN_MATCHES:
                    continue
                good = [m for m, n in (p for p in bf.knnMatch(di, dj, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
                if len(good) < MIN_MATCHES:
                    continue
                p1 = np.float32([ki[m.queryIdx].pt for m in good])
                p2 = np.float32([kj[m.trainIdx].pt for m in good])
                for src in out_dirs:
                    Ki, Kj = Ks[src][i], Ks[src][j]
                    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), Ki, None).reshape(-1, 2)
                    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), Kj, None).reshape(-1, 2)
                    fm = 0.5 * (Ki[0, 0] + Kj[0, 0])
                    E, inl = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 1.0 / fm)
                    if E is None or E.shape != (3, 3) or inl is None:
                        continue
                    n_in, R, _, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=inl.copy())
                    if n_in < MIN_INLIERS or n_in < 0.5 * len(good):
                        continue
                    d = np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ rel_rot(R0[i], R0[j])).as_rotvec()))
                    if d > GROSS_DEG:
                        continue
                    cons[src].append((i, j, *Rotation.from_matrix(R).as_rotvec(), n_in, d))
        info = {}
        for src, od in out_dirs.items():
            c = np.array(cons[src], float).reshape(-1, 7)
            od.mkdir(parents=True, exist_ok=True)
            np.savez(od / f'{k}.npz', cons=c, n_keyframes=len(frames))
            info[src] = dict(constraints=len(c), median_pi3x_diff_deg=round(float(np.median(c[:, 6])), 3) if len(c) else None)
        n_static = float(np.mean([m.mean() for m in masks]))
        return k, dict(split=clip['split'], camera_mode=clip['camera_mode'], keyframes=len(frames),
                       static_frac=round(n_static, 4), pi3x_focal_px=round(float(Ks['pi3x-clip'][0][0, 0]), 1),
                       nominal_focal_px=round(float(K[0, 0]), 1), per_source=info)
    except Exception as exc:
        return k, dict(error=str(exc)[:300])


def refine(R0, ok, cons, sigma_p_deg):
    """Rotation averaging: corrections d_k (rotation vectors, first fixed) of R_k = R0_k exp(d_k)."""
    n = len(R0)
    idx = np.nonzero(ok)[0]
    prior = [(a, b, rel_rot(R0[a], R0[b])) for a, b in zip(idx[:-1], idx[1:])]
    meas = [(int(c[0]), int(c[1]), Rotation.from_rotvec(c[2:5]).as_matrix()) for c in cons]
    pairs = prior + meas
    w = np.array([1 / np.radians(sigma_p_deg)] * len(prior) + [1 / np.radians(SIGMA_F_DEG)] * len(meas))
    free = idx[1:]
    pos = {k: i for i, k in enumerate(free)}
    Rm = np.stack([p[2] for p in pairs])
    A = np.array([p[0] for p in pairs])
    B = np.array([p[1] for p in pairs])

    def rots(x):
        d = np.zeros((n, 3))
        d[free] = x.reshape(-1, 3)
        return R0 @ Rotation.from_rotvec(d).as_matrix()

    def resid(x):
        R = rots(x)
        E = np.transpose(Rm, (0, 2, 1)) @ np.transpose(R[B], (0, 2, 1)) @ R[A]
        return (Rotation.from_matrix(E).as_rotvec() * w[:, None]).ravel()

    J = lil_matrix((3 * len(pairs), 3 * len(free)), dtype=int)
    for r, (a, b) in enumerate(zip(A, B)):
        for k in (a, b):
            if k in pos:
                J[3 * r:3 * r + 3, 3 * pos[k]:3 * pos[k] + 3] = 1
    n_prior = len(prior)
    loss_scale = np.ones(3 * len(pairs))
    res = least_squares(resid, np.zeros(3 * len(free)), jac_sparsity=J, loss='soft_l1', f_scale=3.0, method='trf',
                        x_scale='jac', max_nfev=50)
    R = rots(res.x)
    return R, float(np.degrees(np.abs(res.x).max())) if len(res.x) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--pi3x', type=Path, required=True, help='Pi3X run dir to refine (e.g. runs/pi3x-dev-automerged)')
    ap.add_argument('--crops', type=Path, default=Path('runs/static-regions/crops.json'))
    ap.add_argument('--edits', type=Path, help='structure edits of the Pi3X run (structure_masks.py --auto)')
    ap.add_argument('--output', type=Path, required=True, help='output prefix: <prefix>-sp<sigma>, <prefix>-constraints')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--clips', help='comma-separated subset (debugging)')
    ap.add_argument('--k-sources', default='nominal', help='comma-separated: nominal, pi3x-clip, pi3x-window')
    args = ap.parse_args()
    mp = MANIFESTS[args.split]
    clips = [c for c in select_clips(load_manifest(mp), 'test' if args.split == 'test' else 'train,val', args.clips)
             if c['camera_mode'] in ('wide', 'flir-VIS') and (args.pi3x / 'poses' / f"{c['key']}.npz").exists()]
    crops = json.loads(args.crops.read_text()) if args.crops.is_file() else {}
    edits = json.loads(args.edits.read_text()) if args.edits and args.edits.is_file() else {}
    sources = args.k_sources.split(',')
    prefix = {s: Path(str(args.output) if s == 'nominal' else f'{args.output}-{s}') for s in sources}
    cdirs = {s: Path(f'{prefix[s]}-constraints') for s in sources}
    jobs = [(args.split, c, args.pi3x, float(edits.get(c['key'], {}).get('crop', crops.get(c['key'], 0.0))),
             edits.get(c['key'], {}).get('mask'), cdirs) for c in clips]
    todo = [j for j in jobs if not all((d / f"{j[1]['key']}.npz").is_file() for d in cdirs.values())]
    shared = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, info in ex.map(clip_constraints, todo):
            shared[k] = info
            print(k, info, flush=True)
    for src in sources:
        refine_source(args, clips, prefix[src], cdirs[src], src, shared)


def refine_source(args, clips, output, cdir, src, shared):
    summ_path = cdir / 'summary.json'
    summary = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
    for k, info in shared.items():
        if 'error' in info:
            summary[k] = info
        else:
            summary[k] = dict({x: v for x, v in info.items() if x != 'per_source'}, **info['per_source'][src])
    for c in clips:
        k = c['key']
        cf = cdir / f'{k}.npz'
        if not cf.is_file():
            continue
        cons = np.load(cf)['cons']
        z = load_pose_npz(args.pi3x / 'poses' / f'{k}.npz')
        n = len(z['poses'])
        use = len(cons) >= max(MIN_CONSTRAINTS, MIN_CONSTRAINT_FRAC * n)
        summary.setdefault(k, {})['refined'] = bool(use)
        if not use:
            continue
        R0 = quat_to_rotmat(np.nan_to_num(z['poses'][:, 3:7]) + np.where(np.isfinite(z['poses'][:, 3:7]).all(1, keepdims=True), 0, [0, 0, 0, 1]))
        ok = np.isfinite(z['poses']).all(1)
        for sp in SIGMA_P_GRID:
            R, maxcorr = refine(R0, ok, cons, sp)
            poses = z['poses'].copy()
            poses[ok, 3:] = rotmat_to_quat(R[ok])
            od = Path(f'{output}-sp{sp}') / 'poses'
            od.mkdir(parents=True, exist_ok=True)
            np.savez(od / f'{k}.npz', **dict(z, poses=poses))
            summary[k][f'max_correction_deg_sp{sp}'] = round(maxcorr, 2)
    cdir.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summary, indent=1))
    print(f"{src}: refined {sum(v.get('refined', False) for v in summary.values())}/{len(summary)} clips")


if __name__ == '__main__':
    main()
