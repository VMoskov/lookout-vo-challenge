#!/usr/bin/env python3
"""runs/status_clahe5.md: CLAHE on thermal (fermarble/jobs/clahe5.sh). Reads run outputs only (the heading-only
scores come from oracle.py --saved-heading). --pick-best prints the best dev variant that beats 'current' (lower
mean official drift over the 5 dev flir-IR clips by > 0.05), or nothing."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (_run_heading, heading_change_rms_60s, load_manifest, load_pose_npz, load_timestamps,  # noqa: E402
                    multi_run_heading, select_clips, total_turn_deg)

R = Path('runs')
O = R / 'clahe5'
DEV = ['clip_019', 'clip_022', 'clip_024', 'clip_040', 'clip_046']
VARIANTS = ['current', 'dpvo_med3', 'clahe_med3', 'clahe_med5', 'agree_gate_clahe3', 'agree_gate_clahe5', 'agree_fm_noclahe',
            'agree_gate_clahe5_fm_noclahe', 'fm_always', 'fm_always_noclahe']
AGREE_T, AGREE_D = 10.0, 15.0
SETS = dict(dev=(Path('release/release_manifest.json'), 'train,val', DEV,
                 dict(plain=[f'runs/dpvo-fps8-seed{s}' for s in range(3)], clahe=[f'runs/dpvo-fps8-seed{s}-clahe' for s in range(5)]),
                 dict(vggt='runs/vggt-dev', pi3x='runs/pi3x-dev-cropmerged', vggt_nc='runs/vggt-dev-noclahe', pi3x_nc='runs/pi3x-dev-noclahe')),
            test=(Path('lookout-vo-test-v0.1/release_manifest.json'), 'test', ['clip_100'],
                  dict(plain=[f'runs/dpvo-test-fps8-seed{s}' for s in range(3)], clahe=[f'runs/dpvo-test-fps8-seed{s}-clahe' for s in range(5)]),
                  dict(vggt='runs/vggt-test', pi3x='runs/pi3x-test-cropmerged', vggt_nc='runs/vggt-test-noclahe',
                       pi3x_nc='runs/pi3x-test-crop-noclahe')))


def results(n):
    p = O / f'dev-{n}' / 'results.json'
    return {r['key']: r['drift_pct'] for r in json.loads(p.read_text()) if r['status'] == 'scored'} if p.is_file() else {}


def pick_best():
    cur = results('current')
    if len(cur) < len(DEV):
        return ''
    m0 = np.mean([cur[k] for k in DEV])
    best = min(((np.mean([r[k] for k in DEV]), n) for n in VARIANTS[1:] for r in [results(n)] if len(r) == len(DEV)), default=None)
    return best[1] if best and best[0] < m0 - 0.05 else ''


def heading(path, t):
    p = Path(path)
    if not p.is_file():
        return None
    z = load_pose_npz(p)
    return _run_heading(z['poses'], z['times'], t, 1.0)[0]


def gt_free(L):
    """DPVO turns per seed, med3 / med5, agreement-rule inputs, and FM CLAHE vs no-CLAHE changes (no references)."""
    L += ['## A. DPVO total turn per seed (deg, sigma 1 s) and the agreement rule', '',
          f'Rule: the flir-IR heading is the VGGT/Pi3X circular mean if the DPVO median turns < {AGREE_T:g} deg in total and the '
          f'two models agree (60 s heading-change RMS < {AGREE_D:g} deg); else the DPVO median heading.', '',
          '| clip | plain s0 / s1 / s2 | CLAHE s0 / s1 / s2 / s3 / s4 | plain med3 | CLAHE med3 | CLAHE med5 | FM RMS (CLAHE / no CLAHE) | '
          'rule fires: gate plain3 / CLAHE3 / CLAHE5 (FM CLAHE) | rule fires, FM no CLAHE: plain3 / CLAHE5 |', '|' + '---|' * 9]
    fm_rows = []
    for split, (man, sp, keys, dp, fm) in SETS.items():
        clips = {c['key']: c for c in select_clips(load_manifest(man), sp, None)}
        for k in keys:
            t = load_timestamps(man.parent, clips[k])
            runs = {g: [load_pose_npz(Path(d) / 'poses' / f'{k}.npz') if (Path(d) / 'poses' / f'{k}.npz').is_file() else None
                        for d in ds] for g, ds in dp.items()}
            seed_turn = {g: [total_turn_deg(_run_heading(z['poses'], z['times'], t, 1.0)[0]) if z is not None else np.nan
                             for z in zs] for g, zs in runs.items()}
            med = lambda zs: multi_run_heading([(z['poses'], z['times']) for z in zs if z is not None], t, 1.0)[0]
            turns = dict(plain3=total_turn_deg(med(runs['plain'])), clahe3=total_turn_deg(med(runs['clahe'][:3])),
                         clahe5=total_turn_deg(med(runs['clahe'])) if all(z is not None for z in runs['clahe']) else np.nan)
            h = {n: heading(Path(d) / 'poses' / f'{k}.npz', t) for n, d in fm.items()}
            rms = lambda a, b: heading_change_rms_60s(t, h[a], h[b]) if h[a] is not None and h[b] is not None else np.nan
            r_c, r_n = rms('vggt', 'pi3x'), rms('vggt_nc', 'pi3x_nc')
            fires = lambda g, r: 'yes' if turns[g] < AGREE_T and r < AGREE_D else 'no' if np.isfinite(turns[g]) and np.isfinite(r) else 'n/a'
            f = lambda v: ' / '.join(f'{x:.0f}' for x in v)
            L.append(f"| {k[5:]} ({split}) | {f(seed_turn['plain'])} | {f(seed_turn['clahe'])} | {turns['plain3']:.1f} | {turns['clahe3']:.1f} | "
                     f"{turns['clahe5']:.1f} | {r_c:.1f} / {r_n:.1f} | {fires('plain3', r_c)} / {fires('clahe3', r_c)} / {fires('clahe5', r_c)} | "
                     f"{fires('plain3', r_n)} / {fires('clahe5', r_n)} |")
            # FM change from CLAHE: per model, turn with / without and the 60 s heading-change RMS between them
            row = dict(clip=f'{k[5:]} ({split})')
            for m in ('vggt', 'pi3x'):
                row[m] = (total_turn_deg(h[m]) if h[m] is not None else np.nan,
                          total_turn_deg(h[m + '_nc']) if h[m + '_nc'] is not None else np.nan, rms(m, m + '_nc'))
            hm = lambda a, b: a + np.angle(0.5 * (1 + np.exp(1j * (b - a)))) if a is not None and b is not None else None
            mc, mn = hm(h['vggt'], h['pi3x']), hm(h['vggt_nc'], h['pi3x_nc'])
            row['mean'] = (heading_change_rms_60s(t, mc, mn) if mc is not None and mn is not None else np.nan,
                           float(np.degrees(np.abs(np.angle(np.exp(1j * (mn[-1] - mn[0] - mc[-1] + mc[0])))))) if mc is not None and mn is not None else np.nan)
            fm_rows.append(row)
    L += ['', 'Pi3X / VGGT used here are the runs in use: dev pi3x-dev (via pi3x-dev-cropmerged, uncropped for flir-IR), test pi3x-test-crop '
          '(own-bow crop 0.056); VGGT uncropped.', '']
    L += ['## B. Pi3X and VGGT with vs without CLAHE (GT-free)', '',
          'All existing flir-IR Pi3X / VGGT runs (the ones in every bundle) already had CLAHE on the keyframes (vggt_tool.py since its '
          'first version, same clip limit 2.0 / tiles 8x8 as DPVO), so this compares them with new runs without CLAHE.', '',
          '| clip | VGGT turn CLAHE / no | VGGT RMS diff | Pi3X turn CLAHE / no | Pi3X RMS diff | FM mean: RMS diff / end-heading diff (deg) |',
          '|---|---|---|---|---|---|']
    for r in fm_rows:
        L.append(f"| {r['clip']} | {r['vggt'][0]:.0f} / {r['vggt'][1]:.0f} | {r['vggt'][2]:.1f} | {r['pi3x'][0]:.0f} / {r['pi3x'][1]:.0f} | "
                 f"{r['pi3x'][2]:.1f} | {r['mean'][0]:.1f} / {r['mean'][1]:.1f} |")
    # run-to-run noise: repeat of the CLAHE run on clip_022
    man = SETS['dev'][0]
    c = {x['key']: x for x in select_clips(load_manifest(man), 'train,val', None)}['clip_022']
    t = load_timestamps(man.parent, c)
    noise = []
    for m, a, b in (('VGGT', 'runs/vggt-dev', 'runs/vggt-dev-repeat'), ('Pi3X', 'runs/pi3x-dev', 'runs/pi3x-dev-repeat')):
        ha, hb = heading(Path(a) / 'poses/clip_022.npz', t), heading(Path(b) / 'poses/clip_022.npz', t)
        noise.append(f"{m} {heading_change_rms_60s(t, ha, hb):.2f}" if ha is not None and hb is not None else f'{m} n/a')
    L += ['', f"Run-to-run noise (same CLAHE settings, repeated on clip_022): RMS diff {', '.join(noise)} deg.",
          'RMS diff = RMS over the clip of the difference of the 60 s heading changes (deg).', '']


def main():
    if '--pick-best' in sys.argv:
        print(pick_best())
        return
    L = ['# Status: CLAHE on thermal (flir-IR), 5-seed DPVO and Pi3X / VGGT', '',
         'Generated by `fermarble/jobs/status_clahe5.py` (job `fermarble/jobs/clahe5.sh`). Nothing was uploaded; the current bundle '
         '(w32mean_vwwide) is untouched. Dev flir-IR clips 019, 022, 024, 040 (train) and 046 (val); test clip_100.', '']
    gt_free(L)
    res = {n: results(n) for n in VARIANTS}
    sh = {}
    p = O / 'oracle' / 'saved_heading.csv'
    if p.is_file():
        import csv
        sh = {r['key']: r for r in csv.DictReader(open(p))}
    base = {r['key']: (r['split'], r['drift_pct']) for r in json.loads((R / 'speedfix/dev-vwwide-base/results.json').read_text())
            if r['status'] == 'scored'}
    L += ['## C. Dev: full pipeline (official evaluator) and heading only (oracle: saved heading + true speed)', '',
          'Cells: official drift % (heading-only drift %). flir-IR has constant speed in every variant.', '',
          '| variant | ' + ' | '.join(k[5:] + (' (val)' if base[k][0] == 'val' else '') for k in DEV) +
          ' | flir-IR mean | all-dev train / val mean | better / worse vs current |', '|' + '---|' * (len(DEV) + 4)]
    cur = res['current']
    for n in VARIANTS:
        r = res[n]
        if len(r) < len(DEV):
            L.append(f'| {n} | (missing) |')
            continue
        hd = lambda k: f" ({float(sh[k][f'{n}_head']):.1f})" if k in sh and sh[k].get(f'{n}_head') not in (None, '', 'nan') else ''
        full = {k: (r[k] if k in r else v) for k, (_, v) in base.items()}
        sp = lambda s: np.mean([full[k] for k, (ss, _) in base.items() if ss == s])
        bw = f"{sum(r[k] < cur[k] - 0.05 for k in DEV)} / {sum(r[k] > cur[k] + 0.05 for k in DEV)}"
        L.append(f"| {n} | " + ' | '.join(f'{r[k]:.2f}{hd(k)}' for k in DEV) + f" | {np.mean([r[k] for k in DEV]):.2f} | "
                 f"{sp('train'):.2f} / {sp('val'):.2f} | {bw} |")
    if len(cur) == len(DEV):
        same = all(abs(cur[k] - base[k][1]) < 1e-3 for k in DEV)
        L += ['', f"'current' reproduces the w32mean_vwwide dev results on these clips: {'yes' if same else 'NO'}.",
              "Note: on test, the current pipeline's agreement-rule gate for clip_100 uses the CLAHE DPVO median "
              "(runs/merged-test-seed*-clahe), on dev the plain one ('agree_gate_clahe3' is the test-consistent variant)."]
    L += ['', '## D. Test bundle', '']
    best = pick_best()
    if not best:
        L.append('- none: no variant beats the current flir-IR handling on dev (mean over the 5 flir-IR clips).')
    else:
        bn = R / f'submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_ir-{best}.json'
        chk = (O / f'pack-{best}.log').read_text().strip().splitlines()[-1:] if (O / f'pack-{best}.log').is_file() else ['(no check)']
        L.append(f"- best variant on dev: {best}; bundle `{bn}`" + (f" (sha256 {hashlib.sha256(bn.read_bytes()).hexdigest()})" if bn.is_file() else ' (missing)'))
        L.append(f'- check: {chk[0]}')
        a = json.loads((R / 'submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide.json').read_text())
        b = json.loads(bn.read_text()) if bn.is_file() else None
        if b is not None:
            pa, pb = (np.asarray(x['clip_100']['positions'], float) for x in (a, b))
            ha, hb = (np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))) for p in (pa, pb))
            L.append(f'- clip_100: end-point distance {np.linalg.norm(pa[-1] - pb[-1]) / max(np.linalg.norm(pa[-1] - pa[0]), 1e-9):.2f} x the '
                     f'net displacement; max heading difference {np.degrees(np.abs(np.angle(np.exp(1j * (hb - ha)))).max()):.1f} deg')
    (R / 'status_clahe5.md').write_text('\n'.join(L) + '\n')
    print('written', R / 'status_clahe5.md')


if __name__ == '__main__':
    main()
