# CAtNIPP + SmartMagDL

A runnable adaptation of the original CAtNIPP project that obtains its online
observations from SmartMagDL magnetic simulations instead of Gaussian2D fields.
Python 3.10+; MATLAB is not required for newly generated scenes.

## What is included

- Original CAtNIPP attention encoder/decoder, pointer policy, LSTM, PRM graph,
  PPO rollout format, distributed driver/runner and trajectory-sampling evaluation.
- Python port of SmartMagDL finite-cylinder magnetostatics (Derby/Bulirsch),
  random one/five-magnet scenes, Earth field, smartphone orientation, sensitivity,
  soft-iron distortion, hard-iron offset and additive measurement noise.
- `ideal`, `calibrated` and `raw` sensor modes.
- Optional direct loading of SmartMagDL `TheorB_*.mat` vector fields.
- CPU/CUDA single-process training and evaluation with bounded episode counts,
  checkpoints, trajectory arrays, reconstruction plots and metrics.
- Physics and integration tests; original MATLAB sources for comparison.

Use `--reconstructor gp` or `--reconstructor smartmagdl` to choose the reconstruction
and policy mean input. SmartMagDL uses frozen pretrained weights. Both modes use
the GP covariance as the planner's uncertainty source and reward proxy. The
network outputs one mean map and has no trained uncertainty head.
The original fixed-path densification and dataset-wide map fusion are replaced by
online sampling along the policy's chosen path. No `traj_data.mat` is needed.

## Updated plots, GIFs and SmartMagDL selection

Every `run.py evaluate` episode now saves a final PNG, route-progress PNG frames,
and a GIF by default. `--gif-every 1` records each executed move, `--gif-fps 5`
sets playback speed. `--no-gif` disables frame recording and animation while
retaining the final PNG. Training uses `--plot` for a final image per episode.
The original evaluation entry point also retains PNGs when producing a GIF.

```bash
python run.py evaluate --config configs/calibrated.json --checkpoint runs/magnetic_train/checkpoint.pth --episodes 1 --nodes 200 --budget-m 180 --spacing-m 0.6 --device cuda:1 --reconstructor gp --output runs/jet_evaluation
```

For SmartMagDL, supply the reconstruction weights separately from the CAtNIPP
policy checkpoint and provide its original training scalers. The supported
architecture is `SRResNet` from `Train_SRResNet_smartmagdl.py` in the inspected
repository, including `x = x + res`. That addition is absent from the repository's
inference script. State-dict loading is strict and residual-block count is inferred
from weight keys. Raw state dicts and common checkpoint dictionaries are supported.
Use this command to recover the exact global MinMax extrema from the same MAT
files used to fit scalers in the training script (1000 maps by default).

```bash
python prepare_smartmag_scaling.py --data-dir /path/to/SimA --label-dir /path/to/TheorB --count 1000 --output configs/my_smartmag_scaling.json
python run.py evaluate --config configs/calibrated.json --checkpoint runs/magnetic_train/checkpoint.pth --episodes 1 --nodes 200 --budget-m 180 --spacing-m 0.6 --device cuda:1 --reconstructor smartmagdl --smartmag-checkpoint /path/to/your_SRResNet.pth --smartmag-scaling configs/my_smartmag_scaling.json --output runs/smartmag_evaluation
```

Use the same reconstruction flags with `run.py train` to train/fine-tune the
planner while the reconstruction network is frozen. Match the sensor mode to
the data used to train your reconstruction model. A policy trained with GP means
can load with the SmartMagDL switch, but its inputs change and planning performance
must be assessed or fine-tuned under the selected reconstructor.

The scaling JSON stores `input_min_uT`, `input_max_uT`, `target_min_uT`,
`target_max_uT`, `grid_size` (normally 300), and `interpolation` (default linear).
The original `Bintr` training inputs are in tesla and `B_ground_truth` labels are
in µT. The helper converts the input range to µT and reproduces the two separate
MinMax transforms. It reads training files only. If you changed preprocessing,
provide the matching saved scaling values instead. Weights alone do not contain
these four extrema, so they cannot be recovered from the checkpoint.

Online input maps are interpolated from observations collected so far. Repeated
positions are averaged, and cells outside the observed convex hull are filled
with the median interpolated value. During the initial collinear segment, the
adapter uses median fill and deposits observations in their nearest cells.
SciPy linear interpolation is the default. The MATLAB source uses natural-neighbor
interpolation, so this is a preprocessing difference to validate with your model.
Optional cubic and nearest interpolation are available in the scaling JSON.
Changing interpolation can change reconstruction accuracy.

The selected reconstructor controls policy-node means, RMSE, R², saved prediction
arrays and rendered reconstruction. GP standard deviations remain explicit
uncertainty proxies in SmartMagDL mode. Hypothetical candidate routes use the
frozen observed-data reconstruction while updating GP covariance. They never
query future magnetic readings. Before the first observation, the policy receives
the configured Earth-field baseline.

These settings also work in JSON configurations through `reconstructor`,
`smartmag_checkpoint`, `smartmag_scaling`, `smartmag_device` and `plot_grid_size`.
Config paths are relative to the JSON file. Command-line paths are relative to
the working directory. The original distributed entry points read this JSON via
`SMARTMAG_CONFIG`.

## Budget, termination and learning

Budget is the allowed physical route length. With a 30 m domain, 180 m is 6
normalized distance units. Every segment consumes its Euclidean length. The
policy masks actions that cannot leave enough distance to reach the goal along
the graph. The next episode receives a fresh budget.

An episode terminates on reaching node 0 or the worker's 256-move limit. The
training process ends after `--episodes` episodes. It has no automatic convergence
or early-stopping condition. Eight PPO epochs are performed per episode by
default. `--batch-size 256` is a maximum optimizer minibatch size; an episode with
45 decisions supplies 45 transitions in this single-episode trainer.

The attention policy and critic are learned. The GP is conditioned on each new
measurement using fixed kernel hyperparameters. Selected SmartMagDL weights are
frozen. The reward gives relative GP trace reduction, a -0.1 immediate-revisit
penalty and a terminal penalty equal to minus final trace / 900. The new
`episode_return` field reports their sum.

Convergence would mean that held-out policy performance stabilizes, measured on
fixed scene seeds and budgets. The training script does not test for that. Loss
need not decrease monotonically. Reconstruction RMSE and R² are diagnostics,
while the current policy reward optimizes the uncertainty proxy. A covariance
improvement does not guarantee reconstruction improvement.

## Install

From this directory, preferably in a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Install the PyTorch wheel appropriate for your CUDA installation if using a GPU.
The supplied code defaults to CPU. Example commands below work from the project
root on Windows, Linux or macOS (use your environment's Python command).

## Quick smoke run

```bash
python run.py train --episodes 1 --nodes 40 --budget-m 60 --spacing-m 3 --ppo-epochs 1 --output runs/smoke --plot
python run.py evaluate --checkpoint runs/smoke/checkpoint.pth --episodes 2 --nodes 40 --budget-m 60 --spacing-m 3 --seed 20000 --output runs/eval_smoke --plot
```

A smoke checkpoint only checks execution. It is not a useful trained planner.
The package contains smoke metrics/plots under `validation/`, not pretrained
magnetic weights. Evaluation intentionally requires a checkpoint.

## Train on generated magnetic scenes

```bash
python run.py train --config configs/calibrated.json --episodes 10000 --nodes 200 --budget-m 180 --spacing-m 0.6 --device cuda --output runs/magnetic_train
python run.py evaluate --config configs/calibrated.json --checkpoint runs/magnetic_train/checkpoint.pth --episodes 100 --nodes 200 --budget-m 180 --spacing-m 0.6 --seed 1000000 --device cuda --output runs/magnetic_test --plot
```

Use disjoint seed ranges for training and evaluation. `--checkpoint` during
training loads weights and optimizer, but the supplied `--seed` and episode
counter start a new run; choose a new seed range yourself. Use a new output folder
for each experiment (metrics are appended). `--threads 2` controls CPU PyTorch
threads. Tune graph size, spacing, GP kernel and training duration for the study;
10,000 episodes is an example command, not a convergence guarantee.

`run.py` uses the same PPO clipping (0.2), TD advantages, discounted return
value targets, coefficient 0.5 and gradient clipping 10 as CAtNIPP. It updates
from one complete episode at a time with shuffled minibatches and a constant
learning rate. The distributed upstream driver retains its own batching,
scheduler and worker orchestration.

## Original distributed training and trajectory sampling

```bash
python -m pip install -r requirements-distributed.txt
python driver.py
python eval/test_driver.py
```

Edit `parameters.py` and `eval/test_parameters.py` first. They share the modified
`Env`; there is no Gaussian2D fallback in either active path. Default training
is CPU with two Ray actors; `driver.py` runs until interrupted. Its checkpoint
path is `model/smartmagdl/checkpoint.pth`. `eval/test_driver.py` uses that path.
To use a `run.py` checkpoint, set `model_path` in `eval/test_parameters.py` to its
parent folder. The original `driver.py` resume additionally requires scheduler
and best-performance state, so it cannot directly resume a `run.py` checkpoint.
The `run.py` evaluator is greedy by default. Add `--trajectory-samples 4
--plan-steps 15` to enable multi-candidate lookahead. The original evaluator uses
`TRAJECTORY_SAMPLING`. Candidates are evaluated sequentially on independent
environment copies, avoiding nested multiprocessing inside Ray/CUDA workers.

To select a config in the distributed entry points, set an absolute path before
starting Python:

```bash
# Linux/macOS
export SMARTMAG_CONFIG="$PWD/configs/raw.json"
# PowerShell equivalent:
# $env:SMARTMAG_CONFIG = (Resolve-Path configs/raw.json).Path
```

For multi-node Ray, the code and config/MAT path must be available on every node.
GPU execution and multi-node execution were not validated in the build environment.

## Measurement modes

| Mode | Observation supplied to GP | Intended use |
|---|---|---|
| `ideal` | True total-field magnitude | Isolate planning from sensor errors |
| `calibrated` (default) | Simulate all errors, then invert the known simulated calibration matrix and remove known offset | Calibrated smartphone mapping with noise |
| `raw` | Norm of distorted/noisy phone vector | Reproduce SmartMagDL-style uncalibrated observations |

The calibrated mode assumes calibration parameters are known from calibration,
not inferred by this project. It is not an implementation of real-phone
calibration estimation. In raw mode the field magnitude can depend on heading;
a stationary scalar GP is then an imperfect model. Do not interpret its posterior
standard deviation as a calibrated estimate of all sensor/model errors.

Earth field is `[3,29,33]` µT. Each scene uses 1 or 5 horizontal cylinders, buried
1 m below the observation plane; their centers lie at least 5 m inside a 30 m
square. Diameter is 25–30 mm, length 0.8–1.2 m, and magnetization follows the
source's `1.2706 / mu0 / U(1,2)` A/m. The default shares the first cylinder's
D/L/M among all cylinders, because this is what the MATLAB call actually does.
Set `shared_magnet_properties: false` to use independently sampled properties.

## Units, coordinates and normalization

- Planner `(x,y)` is in `[0,1]^2`; physical position is `(xy - 0.5)*domain_m`.
- Budget and edge lengths inside CAtNIPP are normalized: 180 m / 30 m = 6.
- `run.py` exposes budget and measurement spacing in metres.
- Physics uses metres, A/m and tesla; public measurement magnitudes use µT.
- GP/policy input is `(magnitude_uT - center_uT) / scale_uT`.
- Default center is the Earth-field magnitude, scale is 20 µT. Neither is
  estimated from a hidden test map. Values are not clipped to [0,1].
- GP length scale is normalized spatial distance (0.12 means 3.6 m by default).
  GP alpha is `(gp_noise_uT / scale_uT)^2`, floored at 1e-10.

`ADAPTIVE_AREA=False` defaults to the complete map, including positive and
negative anomalies. Set it to `True` to restore CAtNIPP's upper-confidence-region
objective. Its threshold of 0.4 means +8 µT above `center_uT` with the default
scale. An empty selected region falls back to the full grid. Full-map planning
with fixed GP hyperparameters uses covariance reduction independent of measured
values; measurement-dependent attention is possible, but the reward itself does
not preferentially reward stronger anomalies. Choose the adaptive option if
that is the scientific objective.

The policy sees coordinates, GP mean/std, budget, graph encoding and its history.
Ground truth is used only for offline metrics/plots. Imagined rollout samples
update covariance without calling the simulator's measurement or truth function.

## Use an existing SmartMagDL map

Create a JSON config such as:

```json
{
  "mat_file": "../data/TheorB_1.mat",
  "domain_m": 30.0,
  "sensor_mode": "calibrated",
  "scale_uT": 20.0,
  "gp_noise_uT": 1.0
}
```

Run `python run.py evaluate --config configs/my_map.json ...` with your checkpoint.
Relative MAT paths resolve relative to the config file. Required variables:
`xg`, `yg` (mm), and `Bxt_e`, `Byt_e`, `Bzt_e` (tesla, already including Earth).
Field arrays must have shape `[len(yg),len(xg)]`. No transpose is guessed.
Axes must cover the configured square. Interpolation is bilinear and refuses
out-of-bounds positions. Classic MATLAB MAT files are supported by scipy.loadmat;
MATLAB v7.3/HDF5 files must first be converted to ordinary MAT format.
A single configured MAT file is reused for every episode: use different files
for held-out evaluation. Scalar fused maps cannot reproduce heading-dependent
phone observations and are intentionally not treated as vector ground truth.

## Outputs

- `checkpoint.pth`: model/optimizer state (training only).
- `run_config.json`: command arguments and full magnetic config.
- `metrics.jsonl`: per-episode success, distance, normalized RMSE, RMSE_uT, R²,
  covariance trace and historical MI diagnostic.
- `episode_*.npz`: physical route `[x,y]` in metres about the map center;
  observations `[x_normalized,y_normalized,heading_radians,magnitude_uT]`;
  truth/prediction magnitudes on the 30×30 evaluation grid (x-major flat order).
- `*_samples.png`: dense ground truth, selected reconstruction and GP uncertainty.
- `episode_XXXXX_frames/`: retained route-progress PNGs.
- `episode_XXXXX.gif`: animation created at the end of each evaluation episode.

The upstream `F1Score` name actually denotes regression R²; `run.py` corrects the
output name. Negative R² is possible and should be reported, not hidden.
The inherited `MI = 0.5 logdet(I + 0.01 K)` is a historical covariance diagnostic;
it is not a noise-calibrated magnetic information bound. Raw covariance trace
is in normalized squared units; multiply by `scale_uT**2` for summed µT².
The planner's evaluation grid remains 30×30. Visualization evaluates the simulator
and selected predictor on a separate 300×300 grid, using jet and fixed per-scene
color limits. Change it with --plot-grid-size. The ground truth is freshly
evaluated on that dense grid. Loaded MAT fields retain their native resolution.

## Deliberate fixes and differences from MATLAB/upstream

1. Corrected rotation element (row 2, column 3) to `sa*sb*cr - ca*sr`.
   The source uses `-ca*sb`, which breaks orthogonality.
2. Vertical motion has a valid `atan2` heading; no trajectory points are discarded
   merely because dx=0. Heading is derived from each newly selected segment.
3. Finite-cylinder radial projection handles on-axis points safely.
4. Sampling includes exact spacing boundaries, carries residual distance between
   segments, and supports `None`/zero spacing as endpoint sampling.
5. Actual observations are simulated on demand; repeated visits produce fresh
   noise while the scene's calibration errors remain fixed.
6. Fresh scenes use the full square for magnet placement rather than the bounds
   of a known fixed trajectory. This avoids letting future actions determine the
   hidden scene. No exact MATLAB random-stream equivalence is claimed.
7. Fresh scenes evaluate the analytic field continuously. MATLAB's fixed-path
   nearest-grid sampling plus interpolation/densification is not reproduced.
8. Corrected lookahead LSTM state to the executed prefix; stop at the goal.
9. Removed forced CUDA autocast from the policy for reliable CPU/full precision.
10. Sampling spacing, GP kernel/noise, normalization and full-map defaults are
    explicitly configured for magnetic mapping; the policy architecture is kept.

The GP does not learn a noise model for soft/hard-iron systematic errors. Full
experimental validation, hyperparameter selection and a trained magnetic policy
remain research work. Pretrained Gaussian-field weights were not distributed as
magnetic results.

## Source provenance

Sources inspected on 2026-11-02:

- https://github.com/marmotlab/CAtNIPP, commit
  `eee8c458093697523493665f024a1495fdb1f8c2`
- https://github.com/eladfishergit/smartmagdl, commit
  `77a4c5daad0ec945fde34441f6b618e8b0777024`

CAtNIPP's MIT license is preserved in `LICENSE`. SmartMagDL's README states MIT;
its README and MATLAB files (including author notices) are preserved under
`upstream/`. Original experiment CSVs, demonstration GIFs, large datasets and
Gaussian-trained weights are omitted. They are not needed for this integration.
Cite Cao et al., *Context-Aware Attention-based Network for Informative Path
Planning*, CoRL 2022, and the SmartMagDL repository/publication when reporting
results. See `SOURCE_VERSIONS.json` and `VALIDATION.md`.
