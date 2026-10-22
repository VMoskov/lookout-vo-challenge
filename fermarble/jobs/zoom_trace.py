#!/usr/bin/env python3
"""Any trace of (digital) zoom in the flir-VIS data itself? CPU only, no references.

Per flir-VIS clip of the dev and test manifests:
1. ffprobe -show_format -show_streams: codec, profile, bitrate, encoder, all tags (and any tag
   name mentioning fov/zoom/lens/focal);
2. manifest calibration fields (size, active_rect, fx, fy, provenance);
3. sharpness on N evenly spaced frames, central crop (half width/height, away from the HUD):
   Laplacian variance, and the radial power spectrum of the grey crop as fractions of energy in
   0.25-0.5, 0.5-0.75 and 0.75-1.0 of Nyquist. A 2x digital zoom (upscale) leaves almost no
   energy above 0.5 Nyquist except noise/compression, so hf_ratio = E(0.5-1.0)/E(0.25-0.5) drops.
Writes CSV + text table. Run from the repository root.
"""
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MANIFESTS = [Path('release/release_manifest.json'), Path('lookout-vo-test-v0.1/release_manifest.json')]
KEYWORDS = ('fov', 'zoom', 'lens', 'focal')


def ffprobe(path):
    out = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j['streams'] if s['codec_type'] == 'video')
    tags = {**{f'format.{k}': x for k, x in j['format'].get('tags', {}).items()},
            **{f'stream{s["index"]}.{k}': x for s in j['streams'] for k, x in s.get('tags', {}).items()}}
    return dict(codec=v.get('codec_name'), profile=v.get('profile'), pix_fmt=v.get('pix_fmt'),
                size=f"{v.get('width')}x{v.get('height')}", fps=v.get('r_frame_rate'),
                bitrate_kbps=round(int(j['format'].get('bit_rate', 0)) / 1000),
                n_streams=len(j['streams']), stream_types='/'.join(s['codec_type'] for s in j['streams']),
                side_data='/'.join(sd.get('side_data_type', '') for sd in v.get('side_data_list', [])),
                tags=tags, zoom_tags={k: x for k, x in tags.items() if any(w in k.lower() for w in KEYWORDS)})


def radial_bands(gray):
    g = gray.astype(np.float64)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)        # 1 = Nyquist along each axis
    band = lambda a, b: P[(r >= a) & (r < b)].sum()
    tot = band(0.02, 1.0)
    return band(0.25, 0.5) / tot, band(0.5, 0.75) / tot, band(0.75, 1.0) / tot


def sharpness(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lap, bands = [], []
    for i in np.linspace(0.05, 0.95, n) * total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        c = g[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2]
        lap.append(cv2.Laplacian(c, cv2.CV_64F).var())
        bands.append(radial_bands(c))
    cap.release()
    b = np.median(np.array(bands), axis=0)
    return dict(lap_var=float(np.median(lap)), e_25_50=float(b[0]), e_50_75=float(b[1]), e_75_100=float(b[2]),
                hf_ratio=float((b[1] + b[2]) / b[0]), n_sharp_frames=len(lap))


def clip_row(args):
    root, c, n = args
    path = root / c['video']
    cal = c['calibration']
    row = dict(key=c['key'], split=c['split'], fx=cal['fx'], fy=cal['fy'], width=cal['width'], height=cal['height'],
               container=f"{cal['container_width']}x{cal['container_height']}", active_rect=str(cal['active_rect']),
               provenance=cal['provenance'], file_mb=round(path.stat().st_size / 1e6, 1))
    row.update(ffprobe(path))
    row.update(sharpness(path, n))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--output', type=Path, default=Path('runs/zoom-trace'))
    args = ap.parse_args()
    jobs = []
    for m in MANIFESTS:
        man = json.loads(m.read_text())
        jobs += [(m.parent, c, args.frames) for c in man['clips'] if c['camera_mode'] == 'flir-VIS']
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(clip_row, jobs))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'zoom_trace.json').write_text(json.dumps(rows, indent=1))
    cols = ['key', 'split', 'size', 'container', 'active_rect', 'fx', 'fy', 'provenance', 'codec', 'profile', 'pix_fmt',
            'fps', 'bitrate_kbps', 'stream_types', 'side_data', 'lap_var', 'e_25_50', 'e_50_75', 'e_75_100', 'hf_ratio']
    import csv
    with open(args.output / 'zoom_trace.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols + ['tags', 'zoom_tags'], extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, 'tags': json.dumps(r['tags']), 'zoom_tags': json.dumps(r['zoom_tags'])})
    print('written', args.output / 'zoom_trace.csv')


if __name__ == '__main__':
    main()
