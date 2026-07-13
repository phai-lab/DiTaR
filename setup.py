"""Editable install for DiTaR's RL half.

All packages live at the repo root: `ours` (this work) and the vendored RL
engine + env factories (`rl_engine`, `maniskill_exp`). The
reward model (`ditar_encoder`, under `TSEncoder/`) has its own packaging and is
NOT installed by this file.

If you prefer not to install, `source env.sh` instead (it just sets PYTHONPATH).
"""
from setuptools import setup, find_packages

pkgs = find_packages(where=".", include=[
    "ours", "ours.*",
    "rl_engine", "rl_engine.*",
    "maniskill_exp", "maniskill_exp.*",
])

setup(
    name="ditar-rl",
    version="0.1.0",
    description="WAN-TC latent-diffusion reward + DrQ-v2 RL "
                "(MetaWorld / ManiSkill)",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=pkgs,
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "maniskill_exp": ["*.json"],
        "ours": ["reward_stats/*.json", "configs/**/*.yaml"],
    },
)
