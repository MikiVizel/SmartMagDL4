"""Frozen SmartMagDL reconstruction with a GP uncertainty proxy for CAtNIPP."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import griddata, RegularGridInterpolator
from gp_ipp import GaussianProcessForIPP
from smartmag_network import SRResNet


class FrozenReconstructor:
    def __init__(self, checkpoint_path, scaling_path, device):
        self.device = torch.device(device)
        self.scaling = json.loads(Path(scaling_path).read_text())
        required = ('input_min_uT', 'input_max_uT', 'target_min_uT', 'target_max_uT')
        for key in required:
            if key not in self.scaling or not np.isfinite(self.scaling[key]):
                raise ValueError(f'SmartMagDL scaling JSON requires finite {key}')
        for name in ('input', 'target'):
            if self.scaling[f'{name}_max_uT'] <= self.scaling[f'{name}_min_uT']:
                raise ValueError(f'{name} scaling maximum must exceed minimum')
        self.grid_size = int(self.scaling.get('grid_size', 300))
        if self.grid_size < 2:
            raise ValueError('SmartMagDL grid_size must be at least 2')
        self.interpolation = self.scaling.get('interpolation', 'linear')
        if self.interpolation not in {'linear', 'nearest', 'cubic'}:
            raise ValueError('interpolation must be linear, nearest or cubic')
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        for name in ('state_dict', 'model_state_dict', 'generator_state_dict', 'model'):
            if name in weights and isinstance(weights[name], dict):
                weights = weights[name]
                break
        # import pdb;
        # pdb.set_trace()
        # weights = {key.removeprefix('module.'): value for key, value in weights.items()}
        weights = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in weights.items()}
        blocks = [int(key.split('.')[1]) for key in weights if key.startswith('residual_blocks.')]
        self.model = SRResNet(num_residual_blocks=max(blocks) + 1 if blocks else 16)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def __deepcopy__(self, memo):
        # Frozen weights can be shared by independent candidate environments.
        return self

    def reconstruct(self, image_uT):
        scale = self.scaling
        array = (image_uT - scale['input_min_uT']) / (scale['input_max_uT'] - scale['input_min_uT'])
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)[None, None]
        with torch.inference_mode():
            prediction = self.model(tensor).squeeze(0).squeeze(0).cpu().numpy()
        if prediction.shape != image_uT.shape or not np.all(np.isfinite(prediction)):
            raise ValueError('SmartMagDL returned an invalid map')
        return prediction * (scale['target_max_uT'] - scale['target_min_uT']) + scale['target_min_uT']


@lru_cache(maxsize=4)
def load_reconstructor(checkpoint_path, scaling_path, device):
    return FrozenReconstructor(checkpoint_path, scaling_path, device)


class SmartMagDLForIPP(GaussianProcessForIPP):
    """Network mean at policy nodes/metrics, GP variance for planning.

    Hypothetical samples are added only to the GP covariance calculation.
    The network receives real observations collected so far, with no map truth.
    """
    def __init__(self, node_coords, config):
        super().__init__(node_coords, config)
        if not config.smartmag_checkpoint or not config.smartmag_scaling:
            raise ValueError('SmartMagDL requires --smartmag-checkpoint and --smartmag-scaling. '
                             'Use the input/target MinMax scaling from model training.')
        self.reconstructor = load_reconstructor(str(Path(config.smartmag_checkpoint).resolve()),
                                               str(Path(config.smartmag_scaling).resolve()),
                                               config.smartmag_device)
        self.real_points = []
        self.real_values = []
        self.last_reconstruction_count = 0
        self.interpolator = None
        axis = np.linspace(0, 1, self.reconstructor.grid_size)
        self.xx, self.yy = np.meshgrid(axis, axis)
        self.axis = axis

    def add_observed_point(self, point_pos, value, measurement=True):
        super().add_observed_point(point_pos, value, measurement)
        if measurement:
            self.real_points.append(np.asarray(point_pos).copy())
            self.real_values.append(float(np.asarray(value).item()))

    def update_gp(self):
        super().update_gp()
        if len(self.real_values) == self.last_reconstruction_count:
            return
        # Average repeated visits before scattered interpolation.
        points, indices = np.unique(np.asarray(self.real_points), axis=0, return_inverse=True)
        values = np.bincount(indices, weights=self.real_values) / np.bincount(indices)
        values_uT = values * self.config.scale_uT + self.config.center_uT
        method = self.reconstructor.interpolation
        # A single straight segment cannot define a 2D triangulation.
        if len(points) < 3 or np.linalg.matrix_rank(points - points[0], tol=1e-10) < 2:
            image = np.full(self.xx.shape, np.median(values_uT))
            nearest_x = np.rint(points[:, 0] * (len(self.axis) - 1)).astype(int)
            nearest_y = np.rint(points[:, 1] * (len(self.axis) - 1)).astype(int)
            image[nearest_y, nearest_x] = values_uT
        else:
            image = griddata(points, values_uT, (self.xx, self.yy), method=method)
            finite = np.isfinite(image)
            fill = np.median(image[finite]) if finite.any() else np.median(values_uT)
            image = np.where(finite, image, fill)
        self.input_map_uT = image
        self.prediction_map_uT = self.reconstructor.reconstruct(image)
        normalized = (self.prediction_map_uT - self.config.center_uT) / self.config.scale_uT
        self.interpolator = RegularGridInterpolator((self.axis, self.axis), normalized, bounds_error=True)
        self.last_reconstruction_count = len(self.real_values)

    def predict(self, X, return_std=False):
        points = np.asarray(X).reshape(-1, 2)
        mean = np.zeros(len(points)) if self.interpolator is None else self.interpolator(points[:, [1, 0]])
        if return_std:
            _, std = self.gp.predict(points, return_std=True)
            return mean, std
        return mean


def make_belief(node_coords, config):
    if config.reconstructor == 'smartmagdl':
        return SmartMagDLForIPP(node_coords, config)
    return GaussianProcessForIPP(node_coords, config)
