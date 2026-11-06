import copy

import pytest
import torch
import yaml

from src.utils.data_utils import MultiViewDataset


@pytest.fixture
def dataset():
    generator = torch.Generator().manual_seed(31)
    return MultiViewDataset({'a': torch.randn(12, 8, generator=generator),
                             'b': torch.randn(12, 4, generator=generator)},
                            torch.arange(3).repeat_interleave(4),
                            metadata={'source': 'unit-test', 'is_mock': False})


@pytest.fixture
def config_factory(tmp_path):
    base = {
        'experiment': {'name': 'test', 'dataset': 'mfeat', 'device': 'cpu', 'seed': 7},
        'latent_space': {'info_dim': 3, 'redundancy_factor': 2, 'ecc_type': 'random_gaussian'},
        'model': {'inference_mode': 'semi_amortized', 'encoder_type': 'mlp'},
        'kernels': {'num_mixtures': 2, 'rff_samples': 4},
        'training': {'batch_size': 12, 'epochs': 1, 'lr': 0.001, 'use_amp': False,
                     'use_gp_loss': True, 'alignment_beta': 0.1, 'num_threads': 1,
                     'num_workers': 0, 'progress': False},
    }

    def create(mode='semi_amortized', **changes):
        cfg = copy.deepcopy(base)
        cfg['model']['inference_mode'] = mode
        for section, options in changes.items():
            cfg[section].update(options)
        path = tmp_path / f'config_{len(list(tmp_path.glob("config*")))}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        return cfg, path
    return create
