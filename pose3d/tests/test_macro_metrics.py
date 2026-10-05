"""Macro (class-averaged) metrics and the class labels the cached validation set gets for them."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pose3d.datasets import pascal
from pose3d.datasets.cache import InMemoryDataset
from pose3d.engine.metrics import calculate_evaluation_metrics, macro_metrics, per_class_acc


def _cache(n=6):
    ds = InMemoryDataset.__new__(InMemoryDataset)
    ds.base, ds.img_key, ds.rot_key = None, "img", "rot"
    ds.imgs = torch.zeros(n, 3, 4, 4, dtype=torch.uint8)
    ds.targets = torch.eye(3).repeat(n, 1, 1) * torch.arange(1, n + 1).view(-1, 1, 1)
    ds.store_uint8, ds.n_draws, ds.n = True, 1, n
    ds.clss, ds.cls_key, ds.eval_clss = None, None, None
    return ds


def test_macro_differs_from_micro_when_classes_are_unbalanced():
    # class 0: nine easy samples, class 1: one hard one
    err = np.array([1.0] * 9 + [40.0])
    cls = np.array([0] * 9 + [1])
    assert np.median(err) == 1.0                       # micro: the hard class is invisible
    m = macro_metrics(err, cls)
    assert m["class_mean_median_error"] == pytest.approx(20.5)
    assert m["class_mean_acc@15"] == pytest.approx(0.5)   # (1.0 + 0.0) / 2
    assert m["class_mean_acc@30"] == pytest.approx(0.5)
    assert per_class_acc(err, cls, 15) == {0: 1.0, 1: 0.0}


def test_eval_classes_are_a_separate_key():
    ds = _cache()
    assert "cls_eval" not in ds[0]
    ds.set_eval_classes([0, 0, 1, 1, 2, 2])
    item = ds[3]
    assert item["cls_eval"].item() == 1 and "cls" not in item
    with pytest.raises(ValueError):
        ds.set_eval_classes([0, 1])


def test_metrics_record_eval_classes_without_giving_them_to_the_model():
    seen = []

    class Model(torch.nn.Module):
        def predict(self, x, cls=None, *, n_samples=1):
            seen.append(cls)
            return torch.eye(3).repeat(len(x), 1, 1)

    ds = _cache(4)
    ds.set_eval_classes([0, 0, 1, 1])
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    cfg = SimpleNamespace(device="cpu", run=SimpleNamespace(platform="kaggle"))
    err, cls = calculate_evaluation_metrics(Model(), loader, cfg, return_classes=True)
    assert cls.tolist() == [0, 0, 1, 1] and len(err) == 4
    assert seen == [None, None]


def test_classes_match_cache_detects_a_misordered_annotation_list():
    ds = _cache(4)
    real = [{"rot": ds.targets[i]} for i in range(4)]
    assert pascal._classes_match_cache(ds, real, [0, 0, 1, 1])
    swapped = [real[1], real[0], real[3], real[2]]
    assert not pascal._classes_match_cache(ds, swapped, [0, 0, 1, 1])


def test_attach_eval_classes_skips_instead_of_failing(monkeypatch, capsys, tmp_path):
    cfg = SimpleNamespace(run=SimpleNamespace(path_to_datasets=str(tmp_path)))
    ds = _cache(4)
    pascal._attach_eval_classes(ds, cfg)               # dataset not mounted
    assert ds.eval_clss is None and "not mounted" in capsys.readouterr().out

    real = SimpleNamespace(**{})
    monkeypatch.setattr(pascal, "_class_labels", lambda r: [0, 1, 2])   # wrong length
    pascal._attach_eval_classes(ds, cfg, real=real)
    assert ds.eval_clss is None and "annotations for" in capsys.readouterr().out

    monkeypatch.setattr(pascal, "_class_labels", lambda r: [0, 0, 1, 1])
    pascal._attach_eval_classes(ds, cfg, real=real)
    assert ds.eval_clss.view(-1).tolist() == [0, 0, 1, 1]
