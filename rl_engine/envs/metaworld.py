from metaworld import MT1

try:
    from metaworld import ALL_V3_ENVIRONMENTS as _ALL_METAWORLD_ENVIRONMENTS
except ImportError:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS as _ALL_METAWORLD_ENVIRONMENTS

from .wrapper import (ExtendedTimeStepWrapper, MetaWorldWrapper,
                      TimeLimitWrapper)


def resolve_task_name(task_name):
    if task_name in _ALL_METAWORLD_ENVIRONMENTS:
        return task_name

    if task_name.endswith('-v2'):
        candidate = task_name[:-3] + '-v3'
        if candidate in _ALL_METAWORLD_ENVIRONMENTS:
            return candidate
    elif task_name.endswith('-v3'):
        candidate = task_name[:-3] + '-v2'
        if candidate in _ALL_METAWORLD_ENVIRONMENTS:
            return candidate

    raise KeyError(f'Unknown MetaWorld task: {task_name}')


def is_metaworld_task(task_name):
    try:
        resolve_task_name(task_name)
        return True
    except KeyError:
        return False


def mw_gym_make(task_name, task_id=0, seed=None, camera_name='corner'):
    resolved_task_name = resolve_task_name(task_name)
    if seed is not None:
        mt1 = MT1(resolved_task_name, seed=seed)
    else:
        mt1 = MT1(resolved_task_name)
    env = mt1.train_classes[resolved_task_name](render_mode='rgb_array', camera_name=camera_name)

    if task_id is not None:
        env.set_task(mt1.train_tasks[task_id])
    return env, mt1


def make(name, frame_stack, action_repeat, seed, img_size=64, episode_length=100, task_id=0, camera_name='corner'):
    env, mt1 = mw_gym_make(name, task_id=task_id, seed=seed, camera_name=camera_name)

    env = MetaWorldWrapper(env, img_size, frame_stack, action_repeat, mt1=mt1)
    env = TimeLimitWrapper(env, max_episode_steps=episode_length)
    env = ExtendedTimeStepWrapper(env)

    return env
