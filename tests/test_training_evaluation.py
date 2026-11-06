import math

import pytest
import torch
import yaml

from auto_analyze import analyze_experiment
from src.trainer.engine import Trainer
from src.utils.experiment import INFERENCE_MODES


@pytest.mark.parametrize('mode', INFERENCE_MODES)
def test_all_modes_train_and_evaluate(mode, dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory(mode)
    trainer = Trainer(path, experiment_dir=tmp_path / mode)
    trainer.prepare_data()
    trainer.build_model()
    if hasattr(trainer.model, 'encoder'):
        before = {name: p.detach().clone() for name, p in trainer.model.encoder.named_parameters()}
    trainer.train_epoch(0)
    if hasattr(trainer.model, 'encoder'):
        assert any(not torch.equal(before[name], p) for name, p in trainer.model.encoder.named_parameters())
    trainer.writer.close()
    trainer = Trainer(path, experiment_dir=tmp_path / f'{mode}_saved')
    trainer.fit()
    metrics = analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    again = analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    for key in ('acc', 'nmi', 'robustness_shift_mean', 'alignment_shift_mean'):
        assert metrics[key] == again[key] or math.isnan(metrics[key]) and math.isnan(again[key])
    assert math.isnan(metrics['avg_recon_error'])
    if mode == 'direct':
        assert math.isnan(metrics['robustness_shift_mean'])
    else:
        assert set(metrics['missing_view_metrics']) == set(dataset.views)
    if mode not in ('semi_amortized', 'coded_semi_amortized'):
        assert math.isnan(metrics['alignment_shift_mean'])


def test_seed_reproduces_training_and_checkpoint(dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory('coded_semi_amortized')
    states = []
    for run in ('first', 'repeat'):
        trainer = Trainer(path, experiment_dir=tmp_path / run)
        trainer.fit()
        states.append({key: value.clone() for key, value in trainer.model.state_dict().items()})
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


def test_evaluation_rejects_wrong_config_data_and_weights(dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    cfg, path = config_factory()
    trainer = Trainer(path, experiment_dir=tmp_path / 'run')
    checkpoint_path = trainer.fit()
    saved_config = trainer.ckpt_dir / 'config.yaml'
    cfg['latent_space']['info_dim'] = 4
    saved_config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Config'):
        analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    saved_config.write_text(path.read_text())
    original = dataset.views['a'][0, 0].clone()
    dataset.views['a'][0, 0] += 1
    with pytest.raises(ValueError, match='Dataset'):
        analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    dataset.views['a'][0, 0] = original
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint['model_state_dict'].pop(next(iter(checkpoint['model_state_dict'])))
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match='Missing key'):
        analyze_experiment(trainer.ckpt_dir, dataset=dataset)


def test_trained_reconstruction_is_deterministic(dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory('amortized', training={'use_gp_loss': False})
    trainer = Trainer(path, experiment_dir=tmp_path / 'mse')
    trainer.fit()
    first = analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    second = analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    assert first['avg_recon_error'] == second['avg_recon_error'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_amp_training_step(dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory('coded_semi_amortized', experiment={'device': 'cuda'},
                             training={'use_amp': True})
    trainer = Trainer(path, experiment_dir=tmp_path / 'cuda')
    try:
        trainer.prepare_data()
        trainer.build_model()
        losses = trainer.train_epoch(1)
        assert all(math.isfinite(value) for value in losses)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in trainer.model.encoder.parameters())
    finally:
        trainer.writer.close()


def test_singleton_remainder_keeps_all_samples(dataset, config_factory, monkeypatch, tmp_path):
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory(training={'batch_size': 11})
    trainer = Trainer(path, experiment_dir=tmp_path / 'batches')
    try:
        trainer.prepare_data()
        assert sum(len(indices) for _, _, indices in trainer.dataloader) == len(dataset)
        trainer.build_model()
        assert all(math.isfinite(value) for value in trainer.train_epoch(1))
    finally:
        trainer.writer.close()


def test_unversioned_checkpoint_is_rejected(config_factory, tmp_path):
    _, path = config_factory()
    folder = tmp_path / 'legacy'
    folder.mkdir()
    (folder / 'config.yaml').write_text(path.read_text())
    torch.save({'q_mu': torch.randn(12, 3)}, folder / 'final_model.pth')
    with pytest.raises(ValueError, match='Legacy'):
        analyze_experiment(folder)


def test_repeated_seed_is_not_an_independent_sample(dataset, config_factory, monkeypatch, tmp_path):
    import copy
    from auto_analyze import write_results
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory()
    trainer = Trainer(path, experiment_dir=tmp_path / 'repeat_check')
    trainer.fit()
    metrics = analyze_experiment(trainer.ckpt_dir, dataset=dataset)
    repeat = copy.deepcopy(metrics)
    repeat['checkpoint'] += '.repeat'
    _, summary = write_results([metrics, repeat], tmp_path / 'summary')
    assert summary.iloc[0]['n_seeds'] == 1
    assert summary.iloc[0]['acc_count'] == 1
    assert math.isnan(summary.iloc[0]['acc_std'])
    repeat['nmi'] += 0.1
    with pytest.raises(ValueError, match='inconsistent'):
        write_results([metrics, repeat], tmp_path / 'invalid')
