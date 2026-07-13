"""Small file-backed cache for expensive dataset index construction."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable


def _cache_disabled() -> bool:
    flag = os.environ.get(
        'DITAR_DATASET_INDEX_CACHE_DISABLE',
        os.environ.get('TASKVAE_DATASET_INDEX_CACHE_DISABLE', ''),
    ).strip().lower()
    return flag in {'1', 'true', 'yes', 'y', 'on'}


def _cache_root() -> Path:
    explicit = os.environ.get(
        'DITAR_DATASET_INDEX_CACHE_ROOT',
        os.environ.get('TASKVAE_DATASET_INDEX_CACHE_ROOT', ''),
    ).strip()
    if explicit:
        return Path(explicit)
    cache_root = os.environ.get('CACHE_ROOT', '').strip()
    if cache_root:
        return Path(cache_root) / 'ditar_dataset_index_cache'
    return Path.home() / '.cache' / 'ditar_dataset_index_cache'


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _key_digest(parts: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(parts), sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _load_pickle(path: Path) -> Any:
    with path.open('rb') as handle:
        return pickle.load(handle)


def _atomic_pickle_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_or_build_index_cache(
    *,
    namespace: str,
    key_parts: dict[str, Any],
    builder: Callable[[], Any],
) -> Any:
    """Loads an index cache or builds it once under a cross-process file lock.

    The cache is intentionally opt-out. Set
    `DITAR_DATASET_INDEX_CACHE_DISABLE=1` to force cold construction.
    Legacy `TASKVAE_*` cache env vars remain supported as fallbacks.
    """

    if _cache_disabled():
        print(f'[DatasetIndexCache] disabled namespace={namespace}', flush=True)
        return builder()

    digest = _key_digest({'namespace': namespace, **key_parts})
    cache_dir = _cache_root() / namespace
    cache_path = cache_dir / f'{digest}.pkl'
    lock_path = cache_dir / f'{digest}.lock'

    if cache_path.exists():
        try:
            payload = _load_pickle(cache_path)
            print(f'[DatasetIndexCache] hit namespace={namespace} path={cache_path}', flush=True)
            return payload
        except Exception as error:  # pragma: no cover - defensive corrupt-cache path
            warnings.warn(
                f'Ignoring unreadable dataset index cache {cache_path}: {error!r}',
                RuntimeWarning,
                stacklevel=2,
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with lock_path.open('a+') as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                try:
                    payload = _load_pickle(cache_path)
                    print(f'[DatasetIndexCache] hit-after-lock namespace={namespace} path={cache_path}', flush=True)
                    return payload
                except Exception as error:  # pragma: no cover - defensive corrupt-cache path
                    warnings.warn(
                        f'Ignoring unreadable dataset index cache {cache_path}: {error!r}',
                        RuntimeWarning,
                        stacklevel=2,
                    )
            build_start = time.monotonic()
            payload = builder()
            _atomic_pickle_dump(cache_path, payload)
            print(
                f'[DatasetIndexCache] built namespace={namespace} path={cache_path} '
                f'in {time.monotonic() - build_start:.2f}s',
                flush=True,
            )
            return payload
    except ImportError:  # pragma: no cover - Linux has fcntl
        build_start = time.monotonic()
        payload = builder()
        _atomic_pickle_dump(cache_path, payload)
        print(
            f'[DatasetIndexCache] built-without-lock namespace={namespace} path={cache_path} '
            f'in {time.monotonic() - build_start:.2f}s',
            flush=True,
        )
        return payload


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def dir_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'mtime_ns': int(stat.st_mtime_ns),
    }
