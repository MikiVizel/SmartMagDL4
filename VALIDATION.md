# Validation performed

Validated on CPU with the exact package versions in validation/tested_versions.json.

- All 8 unittest cases passed: elliptic-integral quadrature; analytic axial field;
  independent Biot–Savart surface-current integration; rotation orthogonality and
  noiseless calibration recovery; seed reproducibility and raw heading dependence;
  MAT axis/unit mapping; GP updates, budget and no-lookahead-truth-access;
  spatial sample spacing including vertical segments.
- Completed one CAtNIPP PPO training episode and optimizer update, 40 random nodes,
  60 m budget, 3 m measurement spacing. Finite loss: 0.0783016756.
- Reloaded that checkpoint and completed two held-out greedy episodes (seeds
  20000 and 20001), both reaching the goal within 60 m.
- Completed a trajectory-sampling episode (seed 30000), two candidates and five
  lookahead steps, reaching the goal after 59.4545 m.
- All Python files passed compileall. Inspected a saved reconstruction plot.

These runs verify plumbing and numerical behavior, not trained policy quality.
The smoke evaluation R² values are negative. No accuracy, convergence or advantage
against baseline planners is claimed. The smoke checkpoint is intentionally not
included as a trained magnetic model. Recreate it with the README command.

Not validated: CUDA, end-to-end Ray-distributed execution, long training,
bitwise agreement with MATLAB, or direct use of an external real TheorB dataset
(the MAT loader was tested with a controlled fixture). The local host forbids
multiprocessing communication sockets; candidate lookahead was made sequential
and tested successfully. Distributed entry points remain available for a normal
workstation/cluster. No MATLAB/Octave reference execution was available.

See validation/ for the complete metrics, run configurations, arrays and plots.

Budget regression (2026-11-09). The original graph for seed 12536 raised the insufficient-budget exception. All three disconnected fixed-endpoint graphs in seeds 10000–19999 (12536, 12705, 13181) were connected on their second draw. The corrected training script completed seed 12536 on CPU with 0.703 m left from a 180 m budget. Nine unit tests passed.


## Visualization and model switch validation (2026-11-10)

- Twelve tests passed, including retained PNG/GIF frames, actual dense truth
  evaluation, training-architecture/scaler behavior, and hypothetical-sample
  isolation for the SmartMagDL mean.
- GP greedy evaluation completed at 300×300 rendering resolution. The resulting
  GIF has 10 frames and the run retained 11 PNGs (including the final image).
- A SmartMagDL-mode PPO optimizer step and a two-candidate trajectory-sampling
  evaluation completed using a controlled identity-network fixture. The latter
  produced a 9-frame GIF and retained 10 PNGs. This verifies integration behavior.
- A generated jet reconstruction plot was inspected visually.
- Your own reconstruction checkpoint/scalers were not supplied. Their prediction
  quality and CUDA execution remain unverified. The test fixture is not packaged
  as a trained magnetic model.
