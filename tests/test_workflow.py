from pathlib import Path
import subprocess
import sys
import threading

import pytest

import server_runner


def test_batch_failure_sets_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / 'configs/test'
    configs.mkdir(parents=True)
    (configs / 'failed.yaml').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['server_runner.py', '--config_dir', str(configs), '--max_workers', '1'])
    monkeypatch.setattr(server_runner, 'get_gpu_count', lambda: 0)
    monkeypatch.setattr(server_runner, 'run_experiment', lambda *args: False)
    with pytest.raises(SystemExit) as error:
        server_runner.main()
    assert error.value.code == 1


def test_automatic_analysis_failure_sets_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / 'configs/test'
    configs.mkdir(parents=True)
    (configs / 'success.yaml').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['server_runner.py', '--config_dir', str(configs),
                                     '--max_workers', '1', '--auto_analyze'])
    monkeypatch.setattr(server_runner, 'get_gpu_count', lambda: 0)
    monkeypatch.setattr(server_runner, 'run_experiment', lambda *args: True)
    def fail_analysis(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(server_runner.subprocess, 'check_call', fail_analysis)
    with pytest.raises(SystemExit) as error:
        server_runner.main()
    assert error.value.code == 1


def test_gpu_slots_follow_early_finishing_task(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / 'configs/test'
    configs.mkdir(parents=True)
    for i in range(3):
        (configs / f'{i}.yaml').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['server_runner.py', '--config_dir', str(configs), '--max_workers', '2'])
    monkeypatch.setattr(server_runner, 'get_gpu_count', lambda: 2)
    started = threading.Event()
    release_slow = threading.Event()
    lock = threading.Lock()
    active = set()
    collisions = []
    assignments = {}

    def simulate(cfg, gpu, *_):
        index = int(Path(cfg).stem)
        with lock:
            if gpu in active:
                collisions.append(gpu)
            active.add(gpu)
            assignments[index] = gpu
        try:
            if index == 0:
                started.set()
                assert release_slow.wait(timeout=5)
            elif index == 1:
                assert started.wait(timeout=5)
            elif index == 2:
                release_slow.set()
            return True
        finally:
            with lock:
                active.discard(gpu)
    monkeypatch.setattr(server_runner, 'run_experiment', simulate)
    server_runner.main()
    assert not collisions
    assert assignments[2] == assignments[1]


def test_runner_preserves_parent_gpu_visibility(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4, GPU-example')
    assert server_runner.visible_gpu_id(0) == '4'
    assert server_runner.visible_gpu_id(1) == 'GPU-example'
    with pytest.raises(ValueError, match='outside'):
        server_runner.visible_gpu_id(2)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert server_runner.visible_gpu_id(1) == '1'


def test_generated_script_runs_from_experiment_folder(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / 'create_experiment.py'), '--name', 'cli_check',
        '--dataset', 'mfeat', '--inference', 'coded_semi_amortized', '--ecc_type', 'repetition',
        '--redundancy', '2', '--z_dim', '3', '--epochs', '1', '--device', 'cpu',
        '--batch_size', '2000', '--num-mixtures', '1', '--rff-samples', '4',
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    folder = next((tmp_path / 'experiments').iterdir())
    result = subprocess.run(['bash', 'run.sh'], cwd=folder, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (folder / 'checkpoints/final_model.pth').exists()
    assert (folder / 'checkpoints/metadata.json').exists()
