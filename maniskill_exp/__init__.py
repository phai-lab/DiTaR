from __future__ import annotations

import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
TASK_REGISTRY_PATH = PACKAGE_ROOT / "task_registry.json"


def load_task_registry() -> dict[str, dict]:
    with TASK_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    if not isinstance(registry, dict):
        raise ValueError("ManiSkill task registry must be a JSON object.")
    return registry

