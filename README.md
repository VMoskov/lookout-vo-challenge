# UNIZG FER LABUST / MARBLE solution

This fork contains our solution to the MaCVi 2027 Maritime Visual Odometry challenge (final entry #19444):
see [`fermarble/README.md`](fermarble/README.md) for the method summary, installation, the bit-exact rebuild
of the entry and the full pipeline, and [`factsheet/`](factsheet/) for the factsheet. Everything below this
section is the organizers' starter README, unchanged.

---

# Maritime Visual Odometry

Estimate a boat-mounted camera's trajectory from RGB or thermal video, calibration
and frame timestamps. Water reflections, waves, changing visibility and distant
shorelines make this a challenging setting for visual odometry.

**Full development release — 23 September 2026.** Download all 92 development clips
(68 training, 24 validation), including RGB and thermal video, calibration, frame
timestamps and reference trajectories. Start developing and evaluating locally now.
See the challenge page for the separate 20-clip test download, online submission
availability and competition dates.

[Challenge page](https://macvi.org/workshop/macvi27/challenges/maritime-visual-odometry) ·
[Challenge rules](RULES.md) · [DPVO quick start](QUICKSTART.md) ·
[Release progress](RELEASE_STATUS.md)

Previously named LOOKOUT Visual Odometry. This is a naming update only: the
repository URL, v0.1 archive names, checksums, clip IDs and submission format are
unchanged. Existing downloads and participant submissions remain valid.

## Try the evaluator now

Python 3.11 or 3.12 is sufficient. The synthetic example needs no GPU, video,
checkpoint, account or private data:

```bash
git clone https://github.com/LOOKOUT-AI/lookout-vo-challenge.git
cd lookout-vo-challenge
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-eval.txt
python examples/make_synthetic_example.py --output runs/synthetic
python starter/evaluate.py --dataset runs/synthetic/release_manifest.json --predictions runs/synthetic/run --split val
```

Inspect `runs/synthetic/run/predictions/clip_000.json` for the prediction format and
`runs/synthetic/run/summary.json` for the score and coverage counts. To try your own
trajectory writer, emit the same format and rerun the evaluator. Use a new output
directory when regenerating the example.

This is an artificial trajectory for testing the interface. It has no video and
cannot be used for inference, model training or a real-world performance claim.

## Download the full development dataset

[Download the full archive (6.8 GB)](https://macvi.org/downloads/lookout/v0.1/lookout-vo-dev-v0.1.tar) ·
[SHA-256 checksum](https://macvi.org/downloads/lookout/v0.1/lookout-vo-dev-v0.1.tar.sha256)

No account is required. Verify the archive before extracting it, then verify its
individual files:

```bash
curl -fLO https://macvi.org/downloads/lookout/v0.1/lookout-vo-dev-v0.1.tar
curl -fLO https://macvi.org/downloads/lookout/v0.1/lookout-vo-dev-v0.1.tar.sha256
sha256sum --check lookout-vo-dev-v0.1.tar.sha256
tar -xf lookout-vo-dev-v0.1.tar
(cd release && sha256sum --check CHECKSUMS.sha256)
```

The package contains the complete development selection: 68 training and 24
validation clips, split by recording boat. The manifest lists the final selection
and camera coverage. Each anonymous sequence includes video, decoded frame
timestamps, nominal calibration and separate local-position references. The separate
test participant package contains no references; those remain organizer-private.

For the competition, submit **one JSON prediction bundle, not a model**. The new
`starter/pack_submission.py` validates and combines the per-clip predictions.
Follow [the test submission steps](QUICKSTART.md#pack-a-test-submission), then
select your eligible final entry in the MaCVi dashboard. Team registration, the
daily allowance and coverage requirements are described in [RULES.md](RULES.md).

The inference wrapper uses DPVO; [QUICKSTART.md](QUICKSTART.md) records the pinned
upstream revision, environment and checkpoint checksum. After installing that CUDA
environment, run from this repository with the extracted `release/` directory:

```bash
python starter/infer.py --dataset release/release_manifest.json --clips clip_000 --output runs/demo
python starter/evaluate.py --dataset release/release_manifest.json --clips clip_000 --predictions runs/demo
```

Use a fresh output directory for each run. For validation, replace `--clips clip_000`
with `--split val`. The package includes the corrected starter; prefer this
repository for subsequent fixes and [release notes](RELEASE_STATUS.md).

Allowed inference inputs are monocular video, calibration and frame timestamps.
GPS, IMU, AIS, speed and heading are not inference inputs. Read [RULES.md](RULES.md)
for training rules and the provisional local evaluation metric. Always report
coverage with scores.

## Follow updates and contribute

Watch this repository's releases for dataset announcements. Use
[GitHub issues](https://github.com/LOOKOUT-AI/lookout-vo-challenge/issues) for starter
questions, interface feedback or reproducible bugs, and the
[MaCVi community](https://macvi.org/discord) for discussion.

The starter is developed by Harin Nallaguntla and LOOKOUT contributors. DPVO is a
separate upstream dependency; its code and pretrained weights are not redistributed
here. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Terms

The starter code is licensed under the [MIT License](LICENSE). The dataset uses
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see `release/LICENSE.txt`.
Dataset terms are separate from the code license. Upstream DPVO,
dependency and checkpoint terms apply separately; see [THIRD_PARTY.md](THIRD_PARTY.md).
