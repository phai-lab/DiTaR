"""Task → natural-language instruction lookup for the WAN TC reward.

Sources:
  - ManiSkill: maniskill_exp/task_registry_v2.json
  - MetaWorld: metaworld_v3_language_instructions.json

Used by ``WanTcLatentDiffusionReward`` when ``default_instruction`` is empty.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict


# The task registries live at the repo root: maniskill_exp/ and the MetaWorld
# instructions JSON. Resolve relative to this file, or honour RL_DATA_ROOT.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ.get("RL_DATA_ROOT", _REPO_ROOT))

_MANISKILL_REGISTRY = _DATA_ROOT / "maniskill_exp/task_registry_v2.json"
_METAWORLD_REGISTRY = _DATA_ROOT / "metaworld_v3_language_instructions.json"


@lru_cache(maxsize=1)
def _load_maniskill() -> Dict[str, str]:
    if not _MANISKILL_REGISTRY.exists():
        return {}
    payload = json.loads(_MANISKILL_REGISTRY.read_text())
    return {k: v.get("instruction", "") for k, v in payload.items() if isinstance(v, dict)}


@lru_cache(maxsize=1)
def _load_metaworld() -> Dict[str, str]:
    if not _METAWORLD_REGISTRY.exists():
        return {}
    payload = json.loads(_METAWORLD_REGISTRY.read_text())
    return {entry["task_name"]: entry.get("instruction_en", "") for entry in payload.get("tasks", [])}


def get_instruction(task_name: str) -> str:
    """Return the natural-language instruction for ``task_name``, or '' if unknown.

    Resolution order: ManiSkill → MetaWorld.
    """
    if not task_name:
        return ""
    ms = _load_maniskill()
    if task_name in ms:
        return ms[task_name]
    mw = _load_metaworld()
    if task_name in mw:
        return mw[task_name]
    return ""
