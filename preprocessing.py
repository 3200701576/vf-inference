"""NPZ input handling and the original three-channel preprocessing."""
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def preprocess(rnflt):
    image = np.asarray(rnflt, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError('rnflt must have shape (H, W) or (1, H, W), with H and W >= 32.')
    if not np.isfinite(image).all():
        raise ValueError('rnflt contains NaN or infinite values.')
    small = gaussian_filter(image, sigma=2.0)
    large = gaussian_filter(image, sigma=4.0)
    stacked = np.stack([image, np.abs(image - small), large], axis=0)
    mean = stacked.mean(axis=(0, 1, 2), keepdims=True)
    std = stacked.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    return torch.from_numpy(((stacked - mean) / std).astype(np.float32))


def input_files(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == '.npz':
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.npz'))
        if files:
            return files
    raise ValueError('Input must be an NPZ file or a directory containing NPZ files.')


def load_input(path):
    # Only rnflt is read: labels and any unrelated metadata are not needed.
    with np.load(path, allow_pickle=False) as sample:
        if 'rnflt' not in sample:
            raise ValueError(f'{Path(path).name}: missing rnflt array.')
        return preprocess(sample['rnflt'])
