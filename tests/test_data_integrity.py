import pytest

from src.utils.data_caltech import load_caltech_data
from src.utils.data_100leaves import load_100leaves_data
from src.trainer.engine import Trainer


@pytest.mark.parametrize('loader,filename', [(load_caltech_data, 'Caltech101-7.mat'),
                                            (load_100leaves_data, '100Leaves.mat')])
def test_download_failure_never_creates_mock_data(loader, filename, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError('offline in test')
    monkeypatch.setattr('urllib.request.urlretrieve', fail)
    with pytest.raises(FileNotFoundError, match='real data'):
        loader(data_dir=tmp_path)
    assert not (tmp_path / filename).exists()


def test_trainer_refuses_mock_data(dataset, config_factory, monkeypatch, tmp_path):
    dataset.metadata['is_mock'] = True
    monkeypatch.setattr('src.trainer.engine.get_dataset', lambda *args, **kwargs: dataset)
    _, path = config_factory()
    trainer = Trainer(path, experiment_dir=tmp_path / 'mock')
    try:
        with pytest.raises(ValueError, match='mock'):
            trainer.prepare_data()
    finally:
        trainer.writer.close()


def test_aloi_rotation_cache_preserves_selection(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from src.utils.data_aloi import load_aloi_data
    from src.utils.helpers import dataset_fingerprint
    for object_id in (1, 2):
        folder = tmp_path / 'raw/png4' / str(object_id)
        folder.mkdir(parents=True)
        for angle in (355, 0, 5, 85, 90, 95, 175, 180, 185, 265, 270, 275):
            pixels = np.full((16, 16, 3), (object_id * 20 + angle) % 256, dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f'{object_id}_r{angle}.png')
    first = load_aloi_data(tmp_path, mode='rotation', num_objects=2)
    cached = load_aloi_data(tmp_path, mode='rotation', num_objects=2)
    assert len(first) == 8
    torch.testing.assert_close(first.labels, torch.tensor([0] * 4 + [1] * 4))
    assert dataset_fingerprint(first) == dataset_fingerprint(cached)
    assert not cached.metadata['is_mock']
    subset = load_aloi_data(tmp_path, mode='rotation', num_objects=2, reduce_objects=1)
    assert len(subset) == 4
    assert all(isinstance(value, int) for value in subset.metadata['object_ids'])


def test_aloi_missing_images_are_rejected(tmp_path):
    from src.utils.data_aloi import load_aloi_data
    (tmp_path / 'raw/png4').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='Missing ALOI'):
        load_aloi_data(tmp_path, mode='rotation', num_objects=1)
