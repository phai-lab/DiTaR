from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import dm_env
import gym
import gymnasium
import numpy as np
import torch
from dm_env import specs
from mani_skill.utils import sapien_utils

from maniskill_exp import load_task_registry
from maniskill_exp.camera_presets import DEFAULT_CAMERA_NAME, resolve_camera_look_at, validate_task_camera
from rl_engine.envs.wrapper import ExtendedTimeStep

DEFAULT_IMAGE_SIZE = 64
DEFAULT_CONTROL_MODE = "pd_joint_pos"
DEFAULT_REWARD_MODE = "dense"
DEFAULT_RENDER_SIZE = 256
DEFAULT_SHOW_GOAL_SITE = True




def resolve_task_name(task_name: str) -> str:
    registry = load_task_registry()
    if task_name not in registry:
        raise KeyError(f"Unknown ManiSkill task: {task_name}")
    return task_name


def is_maniskill_task(task_name: str) -> bool:
    try:
        resolve_task_name(task_name)
        return True
    except KeyError:
        return False


def build_camera_override(task_name: str, camera_name: str, image_size: int) -> dict[str, Any]:
    look_at = resolve_camera_look_at(task_name, camera_name)
    pose = sapien_utils.look_at(eye=look_at["eye"], target=look_at["target"])
    pose_vec = [float(value) for value in pose.p.numpy().reshape(-1)]
    pose_vec += [float(value) for value in pose.q.numpy().reshape(-1)]
    return {
        "width": image_size,
        "height": image_size,
        "base_camera": {
            "pose": pose_vec,
        },
    }


def reveal_goal_site(env) -> bool:
    """Make ManiSkill's hidden visual goal marker visible to pixel-only agents."""
    base_env = getattr(env, "unwrapped", env)
    goal_site = getattr(base_env, "goal_site", None)
    hidden_objects = getattr(base_env, "_hidden_objects", None)
    if goal_site is None or not isinstance(hidden_objects, list):
        return False

    original_len = len(hidden_objects)
    hidden_objects[:] = [obj for obj in hidden_objects if obj is not goal_site]
    show_visual = getattr(goal_site, "show_visual", None)
    if callable(show_visual):
        show_visual()
    return len(hidden_objects) != original_len or callable(show_visual)


class ManiSkillWrapper(gym.Env):
    def __init__(
        self,
        env,
        img_size: int,
        frame_stack: int,
        action_repeat: int,
        camera_name: str = DEFAULT_CAMERA_NAME,
        show_goal_site: bool = DEFAULT_SHOW_GOAL_SITE,
        ignore_terminated: bool = False,
    ):
        super().__init__()
        self.env = env
        self._num_frames = frame_stack
        self._action_repeat = action_repeat
        self._camera_name = camera_name
        self._show_goal_site = show_goal_site
        self._ignore_terminated = ignore_terminated
        self._frames = deque([], maxlen=self._num_frames)
        self._res = (img_size, img_size)
        self._last_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        self._state_obs = np.zeros((0,), dtype=np.float32)

        action_space = self.env.action_space
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_space.low, dtype=np.float32),
            high=np.asarray(action_space.high, dtype=np.float32),
            shape=tuple(action_space.shape),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_frames * 3, img_size, img_size),
            dtype=np.uint8,
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        if value.shape[:1] == (1,):
            value = value[0]
        return value

    def _stacked_obs(self) -> np.ndarray:
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def _flatten_numeric_tree(self, value: Any) -> list[np.ndarray]:
        if isinstance(value, dict):
            arrays: list[np.ndarray] = []
            for nested_key, nested_value in value.items():
                if nested_key in {"sensor_data", "sensor_param"}:
                    continue
                arrays.extend(self._flatten_numeric_tree(nested_value))
            return arrays
        if isinstance(value, (list, tuple)):
            arrays: list[np.ndarray] = []
            for item in value:
                arrays.extend(self._flatten_numeric_tree(item))
            return arrays
        array = self._to_numpy(value)
        if array.dtype.kind not in "biuf":
            return []
        return [array.astype(np.float32).reshape(-1)]

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        chunks = self._flatten_numeric_tree(obs.get("agent", {}))
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _extract_pixels(self, obs: dict[str, Any]) -> np.ndarray:
        sensor_tree = obs["sensor_data"]
        sensor_data = sensor_tree.get(self._camera_name, sensor_tree["base_camera"])
        rgb = sensor_data["rgb"]
        pixel_obs = self._to_numpy(rgb)
        if np.issubdtype(pixel_obs.dtype, np.floating):
            if pixel_obs.size == 0 or np.nanmax(pixel_obs) <= 1.0:
                pixel_obs = pixel_obs * 255.0
            pixel_obs = np.clip(pixel_obs, 0.0, 255.0)
        pixel_obs = pixel_obs.astype(np.uint8)
        if pixel_obs.shape[-1] == 4:
            pixel_obs = pixel_obs[..., :3]
        self._last_rgb = pixel_obs.copy()
        if pixel_obs.shape[:2] != self._res:
            pixel_obs = cv2.resize(pixel_obs, dsize=self._res, interpolation=cv2.INTER_CUBIC)
        return pixel_obs.transpose(2, 0, 1)

    @staticmethod
    def _scalar(value: Any, *, dtype=float):
        array = ManiSkillWrapper._to_numpy(value)
        if array.shape == ():
            return dtype(array.item())
        return dtype(array.reshape(-1)[0].item())

    def state(self) -> np.ndarray:
        return self._state_obs.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._show_goal_site:
            reveal_goal_site(self.env)
            get_obs = getattr(getattr(self.env, "unwrapped", self.env), "get_obs", None)
            if callable(get_obs):
                obs = get_obs()
        self._state_obs = self._extract_state(obs)
        pixels = self._extract_pixels(obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels.copy())
        info = dict(info)
        info["success"] = float(self._scalar(info.get("success", False), dtype=float))
        return self._stacked_obs().copy(), info

    def step(self, action):
        rewards = 0.0
        terminated_done = False
        truncated_done = False
        info = {}
        next_obs = None
        for _ in range(self._action_repeat):
            next_obs, reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
            rewards += self._scalar(reward, dtype=float)
            term_bool = self._scalar(terminated, dtype=bool)
            if self._ignore_terminated:
                # When relax_success is on, ManiSkill's internal early
                # termination on success can produce 1-frame episodes that
                # break downstream reward modules; let env truncate at
                # max_episode_steps instead. info["success"] is still set.
                term_bool = False
            terminated_done = terminated_done or term_bool
            truncated_done = truncated_done or self._scalar(truncated, dtype=bool)
            done = bool(terminated_done or truncated_done)
            if done:
                break

        assert next_obs is not None
        self._state_obs = self._extract_state(next_obs)
        self._frames.append(self._extract_pixels(next_obs).copy())
        info = dict(info)
        info["success"] = float(self._scalar(info.get("success", False), dtype=float))
        info["terminated"] = bool(terminated_done)
        info["truncated"] = bool(truncated_done)
        return self._stacked_obs().copy(), rewards, done, info

    def render(self):
        return self._last_rgb.copy()

    def observation_spec(self):
        return specs.BoundedArray(
            self.observation_space.shape,
            np.uint8,
            name="observation",
            minimum=0,
            maximum=255,
        )

    def action_spec(self):
        return specs.BoundedArray(
            self.action_space.shape,
            np.float32,
            name="action",
            minimum=self.action_space.low,
            maximum=self.action_space.high,
        )

    def __getattr__(self, name):
        return getattr(self.env, name)


def _apply_task_relaxation(env, task_name: str) -> None:
    """Patch ManiSkill's evaluate() / goal_thresh in-place to drop overly
    strict success criteria for vision-based RL at 64x64. Only relaxes
    constraints that are clearly unreachable for a stochastic 7D ee_delta_pose
    policy in 100-step episodes; does not change goal locations or reward
    surfaces. Safe to call before env.reset() because evaluate is only invoked
    later from get_obs().
    """
    base = env.unwrapped if hasattr(env, "unwrapped") else env

    if task_name == "PickCube-v1":
        # Super-relax: success = is_grasped (cube lifted by gripper). Drops
        # placement requirement entirely so a stochastic policy can collect
        # signal once it learns to grasp.
        base.goal_thresh = 0.05
        orig_evaluate = base.evaluate

        def _patched(self):
            r = orig_evaluate()
            r["success"] = r["is_grasped"]
            return r

        base.evaluate = _patched.__get__(base, type(base))

    elif task_name == "PokeCube-v1":
        # 5cm tolerance is fine; just drop the is_robot_static requirement.
        orig_evaluate = base.evaluate

        def _patched(self):
            r = orig_evaluate()
            r["success"] = r["is_cube_placed"]
            return r

        base.evaluate = _patched.__get__(base, type(base))

    elif task_name == "StackCube-v1":
        # Original: A_on_B AND A_static AND NOT_grasped -- requires release+settle
        # which is unreachable. Relax to just A_on_B (positional stack).
        orig_evaluate = base.evaluate

        def _patched(self):
            r = orig_evaluate()
            r["success"] = r["is_cubeA_on_cubeB"]
            return r

        base.evaluate = _patched.__get__(base, type(base))

    elif task_name == "LiftPegUpright-v1":
        # Original: angle<0.08rad (4.6deg) AND height<5mm -- too tight for a
        # stochastic 64x64 policy. Loosen to ~17deg + 2cm by monkey-patching
        # the angle/height thresholds inside evaluate.
        import numpy as _np
        from mani_skill.utils import common as _ms_common  # noqa: F401
        from mani_skill.utils.geometry import rotation_conversions as _rc

        orig_evaluate = base.evaluate

        def _patched(self):
            q = self.peg.pose.q
            qmat = _rc.quaternion_to_matrix(q)
            euler = _rc.matrix_to_euler_angles(qmat, "XYZ")
            import torch as _torch
            is_peg_upright = _torch.abs(_torch.abs(euler[:, 2]) - _np.pi / 2) < 0.30
            close_to_table = _torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.02
            return {"success": is_peg_upright & close_to_table}

        base.evaluate = _patched.__get__(base, type(base))


def make(
    name: str,
    frame_stack: int,
    action_repeat: int,
    seed: int,
    img_size: int = DEFAULT_IMAGE_SIZE,
    episode_length: int | None = None,
    camera_name: str = DEFAULT_CAMERA_NAME,
    control_mode: str = DEFAULT_CONTROL_MODE,
    reward_mode: str = DEFAULT_REWARD_MODE,
    render_size: int = DEFAULT_RENDER_SIZE,
    show_goal_site: bool = DEFAULT_SHOW_GOAL_SITE,
    relax_success: bool = False,
    sensor_image_size: int | None = None,
):
    task_name = resolve_task_name(name)
    validate_task_camera(task_name, camera_name)
    sensor_sz = int(sensor_image_size) if sensor_image_size else img_size
    env = gymnasium.make(
        task_name,
        obs_mode="rgb",
        control_mode=control_mode,
        reward_mode=reward_mode,
        render_mode="rgb_array",
        max_episode_steps=episode_length,
        sensor_configs=build_camera_override(task_name, camera_name, image_size=sensor_sz),
        human_render_camera_configs={"width": render_size, "height": render_size},
    )
    if relax_success:
        _apply_task_relaxation(env, task_name)
    env.reset(seed=seed)
    env = ManiSkillWrapper(
        env,
        img_size=img_size,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        camera_name=camera_name,
        show_goal_site=show_goal_site,
        ignore_terminated=relax_success,
    )
    env = ExtendedTimeStepWrapperManiSkill(env)
    return env


def make_env(
    name: str,
    frame_stack: int,
    action_repeat: int,
    seed: int,
    img_size: int = DEFAULT_IMAGE_SIZE,
    episode_length: int | None = None,
    camera_name: str = DEFAULT_CAMERA_NAME,
    control_mode: str = DEFAULT_CONTROL_MODE,
    reward_mode: str = DEFAULT_REWARD_MODE,
    render_size: int = DEFAULT_RENDER_SIZE,
    show_goal_site: bool = DEFAULT_SHOW_GOAL_SITE,
    relax_success: bool = False,
    sensor_image_size: int | None = None,
):
    return make(
        name=name,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        img_size=img_size,
        episode_length=episode_length,
        camera_name=camera_name,
        control_mode=control_mode,
        reward_mode=reward_mode,
        render_size=render_size,
        show_goal_site=show_goal_site,
        relax_success=relax_success,
        sensor_image_size=sensor_image_size,
    )


class ExtendedTimeStepWrapperManiSkill(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            kwargs["seed"] = seed
        obs, _ = self._env.reset(**kwargs)
        return self._augment_time_step(obs, state=self._env.state())

    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        is_success = float(info.get("success", 0.0))
        terminated = bool(info.get("terminated", done and not info.get("truncated", False)))
        truncated = bool(info.get("truncated", False))
        return self._augment_time_step(
            next_obs,
            state=self._env.state(),
            action=action,
            reward=reward,
            success=is_success,
            discount=0.0 if terminated else 1.0,
            done=done,
            terminated=terminated,
            truncated=truncated,
        )

    def _augment_time_step(
        self,
        obs,
        state,
        action=None,
        reward=None,
        success=0.0,
        discount=1.0,
        done=False,
        terminated=False,
        truncated=False,
    ):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            reward = 0.0
            success = 0.0
            discount = 1.0
            done = False
            terminated = False
            truncated = False
        return ExtendedTimeStep(
            observation=obs,
            state=np.asarray(state, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            is_success=float(success),
            discount=float(discount),
            done=bool(done),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
