import os
import sys

_ADROIT_ENVS = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}

# Sibling packages (e.g. ``maniskill_exp``) live next to ``rl_engine`` in the
# repo and may not be pip-installed, so make the repo root importable
# regardless of cwd (hydra changes cwd before the env factory runs).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def make_env(name, frame_stack, action_repeat, seed):
    # This open-source release wires the generic dispatcher for MetaWorld
    # (ManiSkill has its own train script + env factory). The Adroit and
    # LIBERO env modules from the upstream Diffusion Reward repo are not
    # included here.
    import rl_engine.envs.metaworld as metaworld
    if metaworld.is_metaworld_task(name):
        return metaworld.make(name, frame_stack, action_repeat, seed)

    raise NotImplementedError(
        f"task {name!r} is not a recognised MetaWorld task. "
        "This release supports MetaWorld via this shared factory "
        "and ManiSkill via maniskill_exp.rl.envs; Adroit/LIBERO are not bundled."
    )
