import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')

import re
from pathlib import Path

import cv2
import rl_engine.rl.drqv2.utils as utils
import hydra
import numpy as np
import torch
import wandb
from maniskill_exp.rl.envs import make_env
# *** "ours" override: swap the reward-model factory to load WAN TC v1.1 ***
from ours.reward_models import make_rm
from rl_engine.rl.drqv2.logger import Logger
from rl_engine.rl.drqv2.replay_buffer import (AMPBuffer,
                                                     ReplayBufferStorage,
                                                     ReplayCache,
                                                     make_replay_loader,
                                                     replay_storage_has_samples)
from rl_engine.rl.drqv2.video import (LiveWindow, TrainVideoRecorder,
                                             VideoRecorder)
from dm_env import specs

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self._configure_render_backend()
        if self.cfg.use_wandb:
            exp_name = cfg.task_name + '_' + cfg.exp_name + '_' + str(cfg.seed) + '_server1'
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            proj_name = "daw" if cfg.domain == 'adroit' else "daw_" + cfg.domain
            wandb.init(project=proj_name,
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep
        self.use_rm = cfg.use_rm
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # reward model
        if self.use_rm:
            self.cfg.reward.obs_shape = self.train_env.observation_spec().shape
            self.cfg.reward.action_shape = self.train_env.action_spec().shape[0]
            self.rm = make_rm(self.cfg.reward).to(self.device)
            self.replay_cache = ReplayCache(device=self.device)
            self.eval_replay_cache = ReplayCache(device=self.device)
            self.pretrain_rm = self.cfg.reward.pretrain_rm

            if self.cfg.reward.rm_model == 'amp':
                self.amp_buffer = AMPBuffer(batch_size=self.cfg.batch_size, cfg=self.cfg)
        else:
            self.rm = None
            self.pretrain_rm = False
        self.agent.rm = self.rm if self.use_rm else None

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = make_env(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            img_size=self.cfg.image_size,
            episode_length=self.cfg.episode_length,
            camera_name=self.cfg.camera_name,
            control_mode=self.cfg.control_mode,
            reward_mode=self.cfg.env_reward_mode,
            render_size=self.cfg.render_size,
            show_goal_site=self.cfg.show_goal_site,
            relax_success=getattr(self.cfg, 'relax_success', False),
        )
        self.eval_env = make_env(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            img_size=self.cfg.image_size,
            episode_length=self.cfg.episode_length,
            camera_name=self.cfg.camera_name,
            control_mode=self.cfg.control_mode,
            reward_mode=self.cfg.env_reward_mode,
            render_size=self.cfg.render_size,
            show_goal_site=self.cfg.show_goal_site,
            relax_success=getattr(self.cfg, 'relax_success', False),
        )
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'),
                      specs.Array((1, ), np.float32, 'terminated'),
                      specs.Array((1, ), np.float32, 'truncated'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            self._nstep,
            self._discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)
        self.live_window = LiveWindow('drqv2-train',
                                      fps=self.cfg.live_view_fps,
                                      enabled=bool(self.cfg.live_view))

    def _configure_render_backend(self):
        if not self.cfg.live_view:
            os.environ.setdefault('MUJOCO_GL', 'egl')
            return
        if os.environ.get('DISPLAY'):
            os.environ['MUJOCO_GL'] = 'egl'
        else:
            print('live_view requested but DISPLAY is unset; falling back to egl.')
            os.environ['MUJOCO_GL'] = 'egl'

    def _annotate_live_frame(self, frame, label):
        annotated = frame.copy()
        cv2.putText(annotated, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def _maybe_show_live(self, env, observation=None):
        if not self.cfg.live_view:
            return

        mode = getattr(self.cfg, 'live_view_mode', 'side_by_side')
        size = int(self.cfg.live_view_size)

        render_frame = env.render()
        render_frame = cv2.resize(render_frame, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        render_frame = self._annotate_live_frame(render_frame, 'render')

        if observation is not None:
            model_frame = observation.transpose(1, 2, 0)
            model_frame = cv2.resize(model_frame, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
            model_frame = self._annotate_live_frame(model_frame, 'model input')
        else:
            model_frame = render_frame

        if mode == 'render':
            frame = render_frame
        elif mode == 'model':
            frame = model_frame
        else:
            frame = cv2.hconcat([render_frame, model_frame])

        self.live_window.show(frame)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _calc_online_expl_reward(self, prev_time_step, time_step):
        if self.rm is None or not getattr(self.rm, 'use_expl_reward', False):
            return 0.0
        obs = torch.as_tensor(prev_time_step.observation[None], device=self.device)
        next_obs = torch.as_tensor(time_step.observation[None], device=self.device)
        with torch.no_grad():
            return float(self.rm.calc_expl_reward(obs, next_obs).mean().item())

    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        if self.pretrain_rm:
            total_learned_reward = 0.0

        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            episode_sr = False

            time_step = self.eval_env.reset()
            if self.pretrain_rm:
                self.eval_replay_cache.add(time_step)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.pretrain_rm:
                    self.eval_replay_cache.add(time_step)
                self.video_recorder.record(self.eval_env)
                episode_sr = episode_sr or time_step.is_success
                total_reward += time_step.reward
                step += 1

            total_sr += episode_sr
            episode += 1
            
            if self.pretrain_rm:
                learned_return = self.eval_replay_cache.pop(None, self.rm)
                total_learned_reward += learned_return

            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.pretrain_rm: 
                log('episode_learned_reward', total_learned_reward / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward, episode_augmented_reward, episode_sr = 0, 0, 0, False
        time_step = self.train_env.reset()

        if not self.pretrain_rm:
            self.replay_storage.add(time_step)
        else:
            self.replay_cache.add(time_step)

        self.train_video_recorder.init(time_step.observation)
        self._maybe_show_live(self.train_env, time_step.observation)
        metrics = None
        try:
            while train_until_step(self.global_step):
                if time_step.last():
                    self._global_episode += 1
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')

                    # reset env
                    time_step = self.train_env.reset()

                    if not self.pretrain_rm:
                        self.replay_storage.add(time_step)
                    else:
                        learned_return = self.replay_cache.pop(self.replay_storage, self.rm)
                        self.replay_cache.add(time_step)

                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                          ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_success_rate', episode_sr)
                            log('episode_reward', episode_reward)
                            if self.use_rm:
                                if self.pretrain_rm:
                                    expl_return = episode_augmented_reward - episode_reward
                                    logged_augmented_reward = learned_return + expl_return
                                    if self.rm.use_env_reward:
                                        logged_augmented_reward += episode_reward
                                else:
                                    logged_augmented_reward = episode_augmented_reward
                                log('episode_augmented_reward', logged_augmented_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)
                            if self.pretrain_rm:
                                log('episode_learned_reward', learned_return)

                    self.train_video_recorder.init(time_step.observation)
                    self._maybe_show_live(self.train_env, time_step.observation)
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    episode_sr = False
                    episode_step = 0
                    episode_reward = 0
                    episode_augmented_reward = 0

                # try to evaluate
                if eval_every_step(self.global_step):
                    self.logger.log('eval_total_time', self.timer.total_time(),
                                    self.global_frame)
                    self.eval()

                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)

                # try to update the agent
                if (not seed_until_step(self.global_step)
                        and self.global_step % self.cfg.update_every_steps == 0
                        and replay_storage_has_samples(self.replay_storage)):
                    batch = next(self.replay_iter)
                    metrics = dict()

                    if self.use_rm and self.global_step % self.rm.expl_update_interval == 0:
                        if self.cfg.reward.rm_model == 'amp':
                            expert_obs = self.amp_buffer.sample()
                            metrics.update(self.rm.update(batch,  expert_obs))
                        else:
                            metrics.update(self.rm.update(batch))
                    metrics.update(self.agent.update(batch, self.global_step))

                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                # take env step
                prev_time_step = time_step
                time_step = self.train_env.step(action)
                episode_sr = episode_sr or time_step.is_success
                self._maybe_show_live(self.train_env, time_step.observation)
                episode_reward += time_step.reward
                episode_augmented_reward += time_step.reward + self._calc_online_expl_reward(prev_time_step, time_step)

                if not self.pretrain_rm:
                    self.replay_storage.add(time_step)
                else:
                    self.replay_cache.add(time_step)

                self.train_video_recorder.record(time_step.observation)
                episode_step += 1
                self._global_step += 1
        finally:
            self.live_window.close()

    def save_snapshot(self):
        # Atomic write: snapshot.pt.tmp -> snapshot.pt via os.replace, so a
        # mid-write kill cannot leave a corrupted (partially-written) file.
        snapshot = self.work_dir / 'snapshot.pt'
        tmp = self.work_dir / 'snapshot.pt.tmp'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with tmp.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, snapshot)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            # weights_only=False: payload contains pickled DrQV2Agent, timer,
            # ints; PyTorch >=2.6 defaults weights_only=True which can't unpickle
            # arbitrary classes. We trust our own checkpoint.
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='../configs/rl', config_name='default')
def main(cfgs):
    workspace = Workspace(cfgs)
    # Auto-resume on driver-crash recovery: if snapshot.pt exists in the same
    # work_dir (resubmit with same EXP_NAME), restore agent + counters so we
    # lose at most one episode of progress instead of restarting from F=0.
    snapshot = workspace.work_dir / 'snapshot.pt'
    if snapshot.exists():
        try:
            workspace.load_snapshot()
            print(
                f"[auto-resume] restored snapshot from {snapshot} "
                f"(global_step={workspace._global_step}, "
                f"global_episode={workspace._global_episode})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[auto-resume] WARN: failed to load snapshot ({exc!r}); "
                f"starting from F=0",
                flush=True,
            )
    workspace.train()


if __name__ == '__main__':
    main()
