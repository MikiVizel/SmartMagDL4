"""Recover the separate global MinMax ranges used by SmartMagDL training."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing SimA_1.mat etc.')
    parser.add_argument('--label-dir', type=Path, required=True, help='Directory containing TheorB_1.mat etc.')
    parser.add_argument('--count', type=int, default=1000, help='Same number of maps used when fitting training scalers')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interpolation', choices=['linear', 'nearest', 'cubic'], default='linear')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    input_min, target_min = np.inf, np.inf
    input_max, target_max = -np.inf, -np.inf
    expected_shape = None
    for index in range(1, args.count + 1):
        # Exact keys and units in the repository's training pipeline.
        measured = np.asarray(loadmat(args.data_dir / f'SimA_{index}.mat')['Bintr']) * 1e6
        target = np.asarray(loadmat(args.label_dir / f'TheorB_{index}.mat')['B_ground_truth'])
        if measured.shape != target.shape or measured.ndim != 2 or measured.shape[0] != measured.shape[1]:
            raise ValueError(f'Map pair {index} must contain matching square arrays')
        if expected_shape is None:
            expected_shape = measured.shape
        if measured.shape != expected_shape:
            raise ValueError('All training maps must have the same dimensions')
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
            raise ValueError(f'Map pair {index} contains nonfinite data')
        input_min = min(input_min, float(measured.min()))
        input_max = max(input_max, float(measured.max()))
        target_min = min(target_min, float(target.min()))
        target_max = max(target_max, float(target.max()))
    if input_max <= input_min or target_max <= target_min:
        raise ValueError('Training ranges must be nonconstant')
    settings = dict(input_min_uT=input_min, input_max_uT=input_max,
                    target_min_uT=target_min, target_max_uT=target_max,
                    grid_size=expected_shape[0], interpolation=args.interpolation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2))
    print(f'Saved training scaling to {args.output}')


if __name__ == '__main__':
    main()
