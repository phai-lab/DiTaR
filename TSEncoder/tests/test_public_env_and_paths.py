from pathlib import Path

from ditar_encoder.data import dataset_index_cache


def test_dataset_index_cache_prefers_ditar_root_env(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    monkeypatch.setenv("TASKVAE_DATASET_INDEX_CACHE_ROOT", str(legacy))
    monkeypatch.setenv("DITAR_DATASET_INDEX_CACHE_ROOT", str(current))

    assert dataset_index_cache._cache_root() == current


def test_dataset_index_cache_supports_legacy_root_fallback(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    monkeypatch.delenv("DITAR_DATASET_INDEX_CACHE_ROOT", raising=False)
    monkeypatch.setenv("TASKVAE_DATASET_INDEX_CACHE_ROOT", str(legacy))

    assert dataset_index_cache._cache_root() == legacy


def test_dataset_index_cache_default_name_is_ditar(monkeypatch, tmp_path):
    monkeypatch.delenv("DITAR_DATASET_INDEX_CACHE_ROOT", raising=False)
    monkeypatch.delenv("TASKVAE_DATASET_INDEX_CACHE_ROOT", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))

    assert dataset_index_cache._cache_root() == Path(tmp_path) / "ditar_dataset_index_cache"


def test_dataset_index_cache_disable_prefers_ditar_env(monkeypatch):
    monkeypatch.setenv("TASKVAE_DATASET_INDEX_CACHE_DISABLE", "1")
    monkeypatch.setenv("DITAR_DATASET_INDEX_CACHE_DISABLE", "0")

    assert dataset_index_cache._cache_disabled() is False
