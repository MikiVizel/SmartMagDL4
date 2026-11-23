"""Dense field rendering and GIF creation. PNG frames are retained."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_episode(env, route, episode, step, path, test_id=0, coordinates=False):
    size = env.config.plot_grid_size
    if getattr(env, '_plot_grid_size', None) != size:
        axis = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(axis, axis, indexing='ij')
        env._plot_grid = np.column_stack((xx.ravel(), yy.ravel()))
        env._plot_truth = env.underlying_distribution.distribution_function(env._plot_grid)
        env._plot_grid_size = size
    means, stds = [], []
    # Bound GP cross-covariance memory for dense maps.
    for start in range(0, len(env._plot_grid), 1024):
        mean, std = env.gp_ipp.predict(env._plot_grid[start:start + 1024], return_std=True)
        means.append(mean)
        stds.append(std)
    config = env.config
    truth = env._plot_truth * config.scale_uT + config.center_uT
    prediction = np.concatenate(means) * config.scale_uT + config.center_uT
    uncertainty = np.concatenate(stds) * config.scale_uT
    points = np.asarray(route) if coordinates else env.node_coords[np.asarray(route, dtype=int)]
    label = 'SmartMagDL' if config.reconstructor == 'smartmagdl' else 'GP'
    uncertainty_label = 'GP uncertainty proxy (µT)' if label == 'SmartMagDL' else 'GP standard deviation (µT)'
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    low, high = float(truth.min()), float(truth.max())
    if high - low < 1e-9:
        low, high = low - .5, high + .5
    for ax, data, title in zip(axes, (truth, prediction, uncertainty),
                             ('Ground truth |B| (µT)', f'{label} |B| (µT)', uncertainty_label)):
        limits = {'vmin': 0, 'vmax': config.scale_uT} if ax is axes[2] else {'vmin': low, 'vmax': high}
        artist = ax.imshow(data.reshape(size, size).T, origin='lower',
                           extent=(0, config.domain_m, 0, config.domain_m),
                           interpolation='bilinear', cmap='jet', **limits)
        ax.plot(points[:, 0] * config.domain_m, points[:, 1] * config.domain_m, 'w-', linewidth=1.5)
        ax.scatter(points[-1, 0] * config.domain_m, points[-1, 1] * config.domain_m,
                   color='white', edgecolor='black', s=24, zorder=4)
        ax.set(title=title, xlabel='x from lower edge (m)', ylabel='y from lower edge (m)')
        fig.colorbar(artist, ax=ax, shrink=.8)
    distance = (env.budget0 - env.budget) * config.domain_m
    fig.suptitle(f'Distance {distance:.1f} / {env.budget0 * config.domain_m:.1f} m  |  '
                 f'RMSE on evaluation grid {env.RMSE * config.scale_uT:.3f} µT')
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = Path(path) / f'{episode}_{test_id}_{step:04d}_samples.png'
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    env.frame_files.append(str(filename))
    return filename


def save_gif(frame_files, output, fps=5):
    if not frame_files:
        raise ValueError('No frames available for GIF')
    if fps <= 0:
        raise ValueError('GIF frame rate must be positive')
    frames = []
    try:
        for filename in frame_files:
            with Image.open(filename) as image:
                frame = image.convert('RGB')
                frame.thumbnail((1100, 500))
                frames.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=256))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=round(1000 / fps), loop=0, disposal=2, optimize=False)
    finally:
        for frame in frames:
            frame.close()
    return str(output)
