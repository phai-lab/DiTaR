"""Reward-model factory for `ours` — the WAN-TC latent-diffusion reward.

`make_rm(cfg)` is the single entry point the training scripts call; register
additional reward models here against the `calc_reward(imgs) -> [T]` contract.
"""
from .wan_tc_diffusion_reward import WanTcLatentDiffusionReward


def make_rm(cfg):
    rm_model = getattr(cfg, "rm_model", "wan_tc_diffusion")
    if rm_model == "wan_tc_diffusion":
        return WanTcLatentDiffusionReward(cfg)
    raise NotImplementedError(f"ours.reward_models does not know rm_model={rm_model!r}")
