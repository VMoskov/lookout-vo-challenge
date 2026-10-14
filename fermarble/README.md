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

## Environment (GPU)

- DPVO pinned clone: `../../DPVO-859bbbf` (commit 859bbbf, `dpvo.pth` sha256 verified). This is the default `DPVO_DIR`.
- micromamba env `dpvo` (`~/micromamba/envs/dpvo`): Python 3.10, torch 2.5.1+cu121, torch-scatter, opencv-python 4.10.0.84, numpy 1.26.4, conda-forge CUDA 12.1 nvcc/headers.
- DPVO build (system nvcc is CUDA 13, so use the env's 12.1 and gcc-12):

```bash
P=~/micromamba/envs/dpvo; T=$P/targets/x86_64-linux
mkdir -p thirdparty && ln -sfn /usr/include/eigen3 thirdparty/eigen-3.4.0   # Eigen 3.4.0
CUDA_HOME=$P PATH=$P/bin:$PATH CC=gcc-12 CXX=g++-12 NVCC_PREPEND_FLAGS="-ccbin g++-12" \
  TORCH_CUDA_ARCH_LIST=8.9 CPATH=$T/include LIBRARY_PATH=$T/lib:$P/lib \
  $P/bin/python -m pip install --no-build-isolation .
```

## Run (from the repository root)

```bash
PY=~/micromamba/envs/dpvo/bin/python
$PY fermarble/selftest.py
$PY fermarble/run_dpvo.py --split train,val --output runs/dpvo-fps8-seed0          # rerun to resume
$PY fermarble/make_predictions.py --split train,val --poses runs/dpvo-fps8-seed0 --output runs/heading
$PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions runs/heading
$PY fermarble/oracle.py --split train,val --poses runs/dpvo-fps8-seed0 --predictions runs/heading --output runs/oracle
```

## Pose convention

`slam.terminate()` inverts DPVO's internal world-to-camera poses, so rows are camera-to-world
`tx ty tz qx qy qz qw`. Heading scores alone cannot confirm this: inverting a mostly-yawing
pose mirrors the heading, and a 3-D Sim(3) undoes a mirror of a planar path. The oracle therefore
reports `fwd_cos`: the median cosine between DPVO's translation direction and camera +z. It is
about +1 for the right convention.
