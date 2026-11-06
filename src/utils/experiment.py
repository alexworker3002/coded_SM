"""Shared model construction and checkpoint contract for training and evaluation."""

import hashlib
import json

import numpy as np

from src.models.cng_model import CNG_MV_GPLVM

CHECKPOINT_VERSION = 2
SEMI_MODES = ('semi_amortized', 'coded_semi_amortized')
INFERENCE_MODES = ('direct', 'amortized', *SEMI_MODES, 'coded_amortized')


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build_model(config, dataset, device):
    latent = config['latent_space']
    options = config.get('model', {})
    kernels = config['kernels']
    mode = options.get('inference_mode', 'direct')
    if mode not in INFERENCE_MODES:
        raise ValueError(f'Unknown inference mode: {mode}')
    for name, value in (('info_dim', latent['info_dim']),
                        ('redundancy_factor', latent['redundancy_factor']),
                        ('num_mixtures', kernels['num_mixtures']),
                        ('rff_samples', kernels['rff_samples'])):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    ecc_type = latent.get('ecc_type', 'none')
    if ecc_type not in ('none', 'repetition', 'random_gaussian'):
        raise ValueError(f'Unknown ECC type: {ecc_type}')
    if ecc_type == 'none' and latent['redundancy_factor'] != 1:
        raise ValueError('ecc_type=none requires redundancy_factor=1')
    view_dims = {key: int(np.prod(value.shape[1:])) for key, value in dataset.views.items()}
    view_shapes = {key: tuple(value.shape[1:]) for key, value in dataset.views.items()
                   if value.ndim == 4}
    configured_shapes = config.get('view_shapes', config.get('views_shapes', {})) or {}
    for key, shape in configured_shapes.items():
        if key not in view_shapes or tuple(shape) != view_shapes[key]:
            raise ValueError(f'Configured image shape does not match data for {key}')
    use_gp_loss = config['training'].get('use_gp_loss', True)
    return CNG_MV_GPLVM(
        num_data=len(dataset), input_dim=latent['info_dim'], view_dims=view_dims,
        redundancy_factor=latent['redundancy_factor'], use_ecc=ecc_type != 'none',
        ecc_mode=ecc_type, inference_mode=mode,
        encoder_type=options.get('encoder_type', 'mlp'), view_shapes=view_shapes,
        num_mixtures=kernels['num_mixtures'], rff_samples=kernels['rff_samples'],
        ecc_matrix_path=latent.get('matrix_path'), use_readouts=not use_gp_loss,
    ).to(device)
