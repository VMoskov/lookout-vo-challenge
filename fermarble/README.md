# fermarble

DPVO heading + constant-speed prior for Maritime Visual Odometry. Imports the unmodified
`starter/` (tracking, public_dataset, metrics, evaluate) so loading, format checks and scoring
are the official ones.

| File | Role |
|---|---|
| `common.py` | Starter imports, pose geometry, heading + constant-speed model, fallback. Never reads references. |
| `run_dpvo.py` | GPU: `tracking.track()` per clip → `poses/<clip>.npz` (full N×7 camera-to-world poses, frame indices, times) + raw starter-format predictions. Resumable. |
| `make_predictions.py` | CPU: npz → predictions at every released frame (straight-line fallback). Never reads references. |
| `oracle.py` | Train/val only, the only reader of references: scores 9 heading/speed variants with the starter metric. |
| `selftest.py` | Synthetic end-to-end test, no video/GPU: fake poses → run_dpvo (faked tracker) → make_predictions (references hidden) → starter evaluate.py + pack_submission.py → oracle. |

## Installation

The code expects the external checkouts next to the repository (the commits are listed in the factsheet,
`factsheet/main.tex`): `../DPVO-859bbbf`, `../Pi3`, `../vggt`, `../vggt-omega`, `../PanSR`. The four
environments come from the lock files in `envs/` (`dpvo` and `pansr`: micromamba; `pi3` and `vggt`: uv venvs).
DPVO, PanSR and VGGT are not on PyPI, so their entries in the lock files are commented out or Git URLs; install
them from the checkouts as below. Pi3 and vggt-omega are not installed: `pi3_tool.py` and `omega_tool.py` add the
checkouts to `sys.path` (`PI3_DIR`, default `../Pi3`; `VGGT_OMEGA_DIR`, default `../vggt-omega`).

- DPVO (`dpvo` env; Python 3.10, torch 2.5.1+cu121, torch-scatter, opencv-python 4.10.0.84, numpy 1.26.4,
  conda-forge CUDA 12.1 nvcc/headers). Built from the checkout with its CUDA extensions; installed with
  `pip install .` (not editable). The system nvcc is CUDA 13, so use the env's 12.1 and gcc-12; from `../DPVO-859bbbf`:

```bash
P=~/micromamba/envs/dpvo; T=$P/targets/x86_64-linux
mkdir -p thirdparty && ln -sfn /usr/include/eigen3 thirdparty/eigen-3.4.0   # Eigen 3.4.0
CUDA_HOME=$P PATH=$P/bin:$PATH CC=gcc-12 CXX=g++-12 NVCC_PREPEND_FLAGS="-ccbin g++-12" \
  TORCH_CUDA_ARCH_LIST=8.9 CPATH=$T/include LIBRARY_PATH=$T/lib:$P/lib \
  $P/bin/python -m pip install --no-build-isolation .
```

  Weights: `dpvo.pth` from the authors' `models.zip` (`download_models_and_data.sh` in the checkout), placed in
  `../DPVO-859bbbf/` (the default `DPVO_DIR`).
- PanSR (`pansr` env; Python 3.10, torch 2.4.1+cu124): editable install from the checkout, following PanSR's
  README (detectron2 from source at commit b0fe2f0, then PanSR, then its CUDA op); from `../PanSR`:

```bash
PY=~/micromamba/envs/pansr/bin/python
$PY -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@b0fe2f0a5bb6496f65246ef35687a386a9f19202'
$PY -m pip install -e .
( cd pansr/modeling/pixel_decoder/ops && FORCE_CUDA=1 $PY setup.py build install )   # MultiScaleDeformableAttention
```

- VGGT (`vggt` venv): editable install from the checkout,
  `uv pip install --python ~/venvs/vggt/bin/python -e ../vggt`.
- Checkpoints: Pi3X, VGGT-1B and PanSR are downloaded from Hugging Face on first use. The VGGT-Ω checkpoint
  (`facebook/VGGT-Omega`, `vggt_omega_1b_512.pt`) is gated on Hugging Face: the organizers need to request
  access at https://huggingface.co/facebook/VGGT-Omega, download it and set `OMEGA_CKPT` to its path.

The installation was tested on our machine; a clean-machine install has not been verified.

## Rebuilding the final entry

`release/final-entry-model-outputs.tar.gz` holds the model outputs the final entry (#19444, sha256
f5e36116...) was built from: DPVO, Pi3X, VGGT and VGGT-Ω poses, speed-fix speeds, fitted parameters and crops
(16.4 MB, sha256 aa11426667cd3e8eea905b1277b1c43ca3bdafdd1416f151252b37aa1cc5a17d). From the repository root,
with the released test package at `lookout-vo-test-v0.1/`, only the `dpvo` env is needed:

```bash
tar -xzf fermarble/release/final-entry-model-outputs.tar.gz   # -> runs/
OUT=runs/final-entry-v2 bash fermarble/jobs/final_entry.sh    # prints the sha256 and compares it with the entry
```

The full pipeline from the videos is `fermarble/jobs/rerun_test.sh` (about 4 h on one RTX 4070 Ti SUPER; not
bit-exact, because DPVO's CUDA kernels are nondeterministic).

## Run (from the repository root)

```bash
PY=~/micromamba/envs/dpvo/bin/python
$PY fermarble/selftest.py
$PY fermarble/run_dpvo.py --split train,val --output runs/dpvo-fps8-seed0          # rerun to resume
$PY fermarble/make_predictions.py --split train,val --poses runs/dpvo-fps8-seed0 --output runs/heading
$PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions runs/heading
$PY fermarble/oracle.py --split train,val --poses runs/dpvo-fps8-seed0 --predictions runs/heading --output runs/oracle
```

## Heading convention

0 deg is the direction the camera faced at the start: the mean flattened camera forward over the
first 3 s of poses (`START_WINDOW_S`), averaged over a window because DPVO's first frames are noisy.
Positive angles are right turns (clockwise seen from above, about the estimated up, like a compass),
so a heading reads as "turned this much since the start". The oracle's true heading uses the same
sense as a compass bearing (clockwise from north), so the two can be plotted side by side.
Integrated paths are right-handed with bearing 0 along +y and 90 deg along +x. Camera heading and DPVO motion heading share
this frame. The metric's global Sim(3) removes any constant heading offset, so the reference changes
readability only, not scores.

## Pose convention

`slam.terminate()` inverts DPVO's internal world-to-camera poses, so rows are camera-to-world
`tx ty tz qx qy qz qw`. Heading scores alone cannot confirm this: inverting a mostly-yawing
pose mirrors the heading, and a 3-D Sim(3) undoes a mirror of a planar path. The oracle therefore
reports `fwd_cos`: the median cosine between DPVO's translation direction and camera +z. It is
about +1 for the right convention.

## flir-VIS focal length (x2.0 for all clips REJECTED on test)

> **Test result (leaderboard):** `submission_test_flow_focal_v2.json` (x2.0 on all flir-VIS, x1.0
> fallback) scored **18.99 %**, worse than `submission_test_flow.json` (x1.0 med3 + flow, **17.94 %**,
> #18758). A single x2.0 for every flir-VIS clip hurts on test, so **the baseline is the x1.0 bundle
> `runs/submission_test_flow.json`**. The leaderboard suggests the true focal length differs per clip
> by varying factors (zoom); the dev findings below explain why one global factor fits train but not test.

- **Finding.** On flir-VIS, DPVO over-turns (clip_075: 2.0x the true heading change in all seeds).
  Re-running DPVO with the released fx, fy multiplied by 2 (`run_dpvo.py --focal-scale 2.0`)
  fixes it. 8 flir-VIS clips, all train (no val clips exist):
  - focal sweep x1.0 / 1.5 / 2.0 / 2.5 (seed 0): x2.0 best overall;
  - 3-seed median confirmation, mean drift of DPVO heading + constant speed: 17.15 -> 7.74 %
    (heading + true speed 12.21 -> 3.75 %); 6 clips better, 001 and 090 worse.
  Report: `runs/focal-report/focal_report.txt`.
- **No ground-truth-free per-clip detector works** (each tested on the 8 train clips):
  1. calibration, resolution, crop, encoding, overlays or the visible bow: nothing separates 001/090;
  2. DPVO mean reprojection residual (`--dpvo-diag`): repeatable across seeds (SD ~0.09 px) but
     picks the better of x1.0/x2.0 on 5/8; consistently wrong on the two clips with the burned-in
     HUD gauge (075, 082);
  3. near/far flow-band agreement at x1.0 vs x2.0: 4/8 (chance);
  4. GeoCalib single-image calibration: 2/8, systematically opposite (075/082 ~60 deg = nominal,
     001/090 ~40 deg). Results: `runs/focal-report/zoom_analysis.json`, `runs/geocalib/geocalib.csv`.
- **Pipeline verified.** Decoded size = manifest container (1920x1080); the starter's crop,
  half-resolution and multiple-of-16 trim scale fx, fy, cx, cy consistently; DPVO's implied HFOV equals
  the nominal one (63.7 deg flir-VIS). Only quirk: flir-VIS fy is 8 % larger than fx (starter config).
  No factor-2 error anywhere.
- **Cause unresolved.** Not a pipeline bug, not camera pan (the HUD gauge shows the camera fixed
  forward), and GeoCalib does not support optical zoom on 075/082. The x2.0 choice rests on the
  train heading results only.
- **Test bundles (not the baseline, see the test result above).** `runs/submission_test_flow_focal.json`: x2.0 on all 11 flir-VIS test clips.
  `runs/submission_test_flow_focal_v2.json` (`fermarble/jobs/build_focal_v2.py`): same, plus a
  ground-truth-free fallback to x1.0 when every x2.0 seed of a clip has NaN poses and an x1.0 seed is
  intact (clip_096, clip_104).

## Rejected options (dev evaluation)

- `run_dpvo.py --patch-strategy GRADIENT_BIAS` (DPVO's textured-patch sampling): on the 67 wide clips within
  seed noise (heading-only mean 9.25 -> 9.43 %, median 4.68 -> 3.62 %) with a few large failures
  (clip_017 12 -> 73 %); worse on flir-VIS and no effect on the 2x over-turning. Not used.
- flir-IR focal x0.5 / x0.75: the low average turn ratio is tracking failure on 3 of 5 clips, not
  calibration; x1.0 is best on train. flir-IR stays x1.0. Reports: `runs/gb-wide/`, `runs/gb-flirvis/`,
  `runs/flirir-focal/`.
