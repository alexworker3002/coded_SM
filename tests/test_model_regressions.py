import numpy as np
import pytest
import torch

from src.modules.ecc import LinearECCProjection
from src.modules.ecc_decoder import LinearECCDecoder
from src.kernels.ng_sm_kernel import NextGenSpectralMixtureKernel
from src.utils.experiment import build_model


@pytest.mark.parametrize('mode', ['repetition', 'random_gaussian'])
def test_projection_roundtrip_and_variance(mode):
    torch.manual_seed(7)
    projection = LinearECCProjection(3, 4, mode)
    decoder = LinearECCDecoder(projection.G)
    z = torch.randn(16, 3)
    recovered, log_sigma = decoder(projection(z), torch.zeros(16, 12))
    torch.testing.assert_close(recovered, z, atol=2e-6, rtol=2e-6)
    noise = torch.randn(30000, 12) @ decoder.G_pinv
    torch.testing.assert_close(noise.var(0), (2 * log_sigma[0]).exp(), atol=0.02, rtol=0.03)
    if mode == 'repetition':
        torch.testing.assert_close((2 * log_sigma).exp(), torch.full_like(log_sigma, 0.25))


def test_invalid_matrix_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        LinearECCProjection(3, 2, matrix_path=tmp_path / 'missing.pkl')
    with pytest.raises(ValueError):
        LinearECCDecoder(torch.zeros(3, 6))


def test_conditional_frequency_scale_and_deterministic_evaluation():
    torch.manual_seed(9)
    kernel = NextGenSpectralMixtureKernel(3, num_mixtures=2, rff_samples=32)
    kernel.eval()
    x = torch.randn(10, 3)
    first = kernel.get_rff_feature(x)
    torch.testing.assert_close(first, kernel.get_rff_feature(x), atol=0, rtol=0)
    first.square().sum().backward()
    # At rho=0, the second marginal's scale must still affect omega2.
    assert kernel.log_std2.grad.abs().sum() > 0


@pytest.mark.parametrize('n,d', [(9, 3), (3, 9)])
def test_gp_marginal_matches_dense_covariance(n, d, config_factory, dataset):
    cfg, _ = config_factory('direct')
    model = build_model(cfg, dataset, 'cpu')
    torch.manual_seed(11)
    phi = torch.randn(n, d) * 0.4
    y = torch.randn(n, 2)
    sigma = torch.tensor(0.7)
    actual = model._compute_gp_marginal_nll(phi, y, sigma)
    covariance = phi @ phi.T + (sigma.square() + 1e-5) * torch.eye(n)
    expected = 0.5 * ((y * torch.linalg.solve(covariance, y)).sum()
                      + y.shape[1] * torch.linalg.slogdet(covariance).logabsdet
                      + n * y.shape[1] * np.log(2 * np.pi))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_image_dimensions_are_flattened_and_shapes_inferred(config_factory):
    from src.utils.data_utils import MultiViewDataset
    images = MultiViewDataset({'image': torch.randn(8, 3, 16, 16)}, torch.arange(8))
    cfg, _ = config_factory('amortized', model={'encoder_type': 'cnn2d'})
    model = build_model(cfg, images, 'cpu')
    model.eval()
    mu, _ = model.infer_latents(images.views)
    assert mu.shape == (8, 3)
    assert model.view_dims['image'] == 3 * 16 * 16
    assert model.readouts['image'] is None
