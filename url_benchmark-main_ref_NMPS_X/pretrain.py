import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

def homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device):
    #print("x_t : ", x_t)
    #print('time_step : ', time_step)
    #print('x_bar : ', x_bar)
    #print('x_squared_bar : ', x_squared_bar)
    #print('x_plus_bar : ', x_plus_bar)
    #print('rho : ', rho)
    Tau = np.minimum(time_step, 100/rho)
    #print("Tau : ", Tau)
    x_bar = (1 - 1/Tau) * x_bar + 1/Tau * x_t
    #print("x_bar : ", x_bar)
    x_squared_bar = (1 - 1/Tau) * x_squared_bar + 1/Tau * ((x_t - x_bar) ** 2)
    #print("x_squared_bar : ", x_squared_bar)
    x_plus = np.exp((x_t - x_bar)/ np.square(x_squared_bar))
    #print("x_plus : ", x_plus)
    x_plus_bar = (1 - 1/Tau) * x_plus_bar + 1/Tau * x_plus
    ##y_t = torch.bernoulli(np.minimum(1, rho*x_plus/x_plus_bar))
    #print("rho*x_plus/x_plus_bar : ", rho*x_plus/x_plus_bar)
    y_t_input = np.minimum(1, rho*x_plus/x_plus_bar)
    #print("y_t_input : ", y_t_input)
    y_t = torch.bernoulli(torch.tensor(y_t_input, dtype=torch.float32).to(device))
    #print("y_t : ", y_t)
    #print("homeostasis_y_t : ", y_t.cpu().numpy())
    return x_bar, x_squared_bar, x_plus_bar, y_t

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        ###################################################################
        # create exploration agent
        self.agent_exploration = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        self.action_m = 0
        self.explore_count = 1
        self.off_policy_count = 0
        self.exploration_policy_count = 0
        self.explore_flag = False
        self.exploit_count = 1
        self.x_bar, self.x_squared_bar, self.x_plus_bar = 0, 1, 1
        ##self.target_rate_rho = 0.01  #### 0.1, 0.01, 0.001, 0.0001
        ##self.target_rate_rho = 0.0001  #### 0.1, 0.01, 0.001, 0.0001
        ##self.target_rate_rho = 0.001  #### 0.1, 0.01, 0.001, 0.0001
        self.target_rate_rho = 0.1  #### 0.1, 0.01, 0.001, 0.0001
        self.explore_fixed_steps = 100		
        #self.explore_fixed_steps = 200
        ##self.update_timestep = 30
        self.update_timestep = 30
        self.old_value, self.old_m_value, self.old_h_target_value, self.old_l_target_value = 0, 0, 0, 0
        self.value_h_reward, self.value_h_reward_m, self.value_l_reward = 0, 0, 0
        self.gamma_tigeer = 1
        self.gamma = 0.99  # discount factor
        ###################################################################


        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

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

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        ##############################################################
        starting_exploration_until_step = utils.Until(self.cfg.num_starting_exploration_frames,
                                       self.cfg.action_repeat)
        ##############################################################

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                    ##############################################################
                    self.save_explor_snapshot()
                    ##############################################################
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            #####################################################################
            if starting_exploration_until_step(self.global_step):
                #print("if starting_exploration_until_step(self.global_step) ", self.global_step)
                #print("starting_exploration")
                self.action_m = 0
                #self.explore_count += 1
                #if starting_exploration_until_step(self.explore_count):
                #    self.explore_count = 1
            else:
                #print("if starting_exploration_else ", self.global_step)
                if self.explore_flag == True:
                    #print("if self.explore_flag == True:")
                    if self.explore_count % self.explore_fixed_steps == 0:
                        #print("self.explore_count : ", self.explore_count)
                        #self.explore_count = 0
                        self.explore_count = 1
                        self.x_bar, self.x_squared_bar, self.x_plus_bar = 0, 1, 1
                        self.explore_flag = False
                    else:
                        #print("if self.explore_count_else")
                        #print("self.explore_count : ", self.explore_count)
                        self.explore_count += 1

                if self.explore_flag == False:
                    #print("if self.explore_flag == False:")
                    if self.exploit_count % self.update_timestep == 0:
                        #next_goal_t = torch.min(torch.max(actor_target_h(next_state), -max_goal), max_goal)
                        #q_target_1, q_target_2 = critic_target_h(next_state, next_goal_t)
                        #q_target_h = torch.min(q_target_1, q_target_2)
                        #q_target_h_n = q_target_h.detach().cpu().numpy()[0][0]

                        ##q_target_h_n = self.agent.read_target_value(self.replay_iter, self.global_step)
                        q_target_h_n = self.agent.read_target_value(time_step.observation,
                                                                    meta,
                                                                    self.global_step)
                        #############q_target_h_n = self.agent_exploration.read_target_value(self.replay_iter, self.global_step)
                        value_h_promise_discrepancy = self.old_h_target_value - self.value_h_reward - (self.gamma_tigeer * q_target_h_n)
                        #print("value_h_promise_discrepancy : ", value_h_promise_discrepancy)
                        abs_value_h_promise_discrepancy = np.absolute(value_h_promise_discrepancy.cpu().numpy())
                        ##value_h_promise_discrepancy = apply_normalizer(abs_value_h_promise_discrepancy, value_h_promise_discrepancy_normalizer)

                        ##record_logger(args=[value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                        ##record_logger(args=[abs_value_h_promise_discrepancy], option='only_h_values_variance', step=t-start_timestep)
                        self.old_h_target_value = q_target_h_n
                        self.value_h_reward = 0
                        self.gamma_tigeer = 1

                        ##homeostasis(x_t, time_step, x_bar, x_squared_bar, x_plus_bar, rho, device)
                        #print("homeostasis")
                        x_bar, x_squared_bar, x_plus_bar, y_t = homeostasis(abs_value_h_promise_discrepancy, self.exploit_count, self.x_bar, self.x_squared_bar, self.x_plus_bar, self.target_rate_rho, self.device)
                        ##print("y_t : ", y_t)
                        y_t_n = y_t.item()
                        #print("y_t_n : ", y_t_n)
                        ##if y_t_n == 1:
                        if y_t_n == 0:
                            #print("if y_t_n == 0:")
                            self.action_m = 1
                            ##exploit_count += 1
                        else:
                            #print("self.exploit_count = 1")
                            self.action_m = 0
                            self.exploit_count = 1
                            self.explore_flag = True

                    else:
                        #print("if explore_flag == False: if y_t == 1: else_222 ")
                        ##########################################################################
                        ############temp_episode_reward_h = episode_reward_h.cpu().numpy()[0]
                        #############temp_episode_reward_h = intr_sf_reward.mean().item()
                        #############temp_episode_reward_h = intr_ent_reward.mean().item()
                        ##########################################################################
                        #intr_sf_reward = self.agent.compute_intr_sf_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                        #intr_ent_reward = self.agent.compute_intr_ent_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                        obs = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(0)
                        h = self.agent.encoder(obs)
                        intr_sf_reward = self.agent.compute_intr_sf_reward(torch.as_tensor([meta['task']],device=self.device), h, self.global_step)
                        ##intr_ent_reward = self.agent_exploration.compute_intr_ent_reward(meta['task'], h, self.global_step)
                        temp_episode_reward_h = intr_sf_reward
                        temp_episode_reward_h = self.gamma_tigeer * temp_episode_reward_h
                        self.value_h_reward += temp_episode_reward_h
                        self.gamma_tigeer = self.gamma_tigeer * self.gamma
                        #print("self.exploit_count += 1")
                        self.exploit_count += 1

            ######################################################################

            # sample action
            if self.action_m == 1:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    #print("if self.action_m == 1 self.global_step : ", self.global_step)
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)
                    #print("if self.action_m == 1 action : ", action)
                    self.off_policy_count += 1
            else:
                with torch.no_grad(), utils.eval_mode(self.agent_exploration):
                    #print("if self.action_m == 1_else self.global_step : ", self.global_step)
                    action = self.agent_exploration.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)
                    #print("if self.action_m == 1_else action : ", action)
                    self.exploration_policy_count += 1

            # try to update the agent
            if not seed_until_step(self.global_step):
                #####################################################################
                #print("if not seed_until_step(self.global_step): ", self.global_step)

                if self.global_step % self.agent.update_every_steps != 0:
                    #print("train_extra")
                    metrics = self.agent.update_pretraining_first(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')


                    metrics = self.agent_exploration.update_pretraining_first(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train_expl')
                else:
                    #print("train_main")
                    replay_buffer_obs, replay_buffer_action, replay_buffer_extr_reward, replay_buffer_discount, replay_buffer_next_obs, replay_buffer_task, first_metrics = self.agent.update_pretraining_first(self.replay_iter, self.global_step)
                    #print("replay_buffer_obs, replay_buffer_action, replay_buffer_extr_reward, replay_buffer_discount, replay_buffer_next_obs, replay_buffer_task, first_metrics : ", replay_buffer_obs, replay_buffer_action, replay_buffer_extr_reward, replay_buffer_discount, replay_buffer_next_obs, replay_buffer_task, first_metrics)
                    ##replay_buffer_obs, replay_buffer_action, replay_buffer_extr_reward, replay_buffer_discount, replay_buffer_next_obs, replay_buffer_task, first_metrics = self.agent_exploration.update_pretraining_first(self.replay_iter, self.global_step)
                    #################self.agent.direct_update_aps(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    intr_sf_reward = self.agent.compute_intr_sf_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    intr_ent_reward = self.agent.compute_intr_ent_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    metrics = self.agent.update_pretraining_second(self.replay_iter,
                                                                   self.global_step,
                                                                   replay_buffer_obs,
                                                                   replay_buffer_action,
                                                                   replay_buffer_extr_reward,
                                                                   replay_buffer_discount,
                                                                   replay_buffer_next_obs,
                                                                   replay_buffer_task,
                                                                   first_metrics,
                                                                   intr_ent_reward,
                                                                   intr_sf_reward,
                                                                   True,
                                                                   self.off_policy_count,
                                                                   self.exploration_policy_count)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')


                    ##############replay_buffer_obs, replay_buffer_action, replay_buffer_extr_reward, replay_buffer_discount, replay_buffer_next_obs, replay_buffer_task, first_metrics = self.agent_exploration.update_pretraining_first(self.replay_iter, self.global_step)
                    self.agent_exploration.direct_update_aps(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    intr_sf_reward = self.agent_exploration.compute_intr_sf_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    intr_ent_reward = self.agent_exploration.compute_intr_ent_reward(replay_buffer_task, replay_buffer_next_obs, self.global_step)
                    metrics = self.agent_exploration.update_pretraining_second(self.replay_iter,
                                                                               self.global_step,
                                                                               replay_buffer_obs,
                                                                               replay_buffer_action,
                                                                               replay_buffer_extr_reward,
                                                                               replay_buffer_discount,
                                                                               replay_buffer_next_obs,
                                                                               replay_buffer_task,
                                                                               first_metrics,
                                                                               intr_ent_reward,
                                                                               intr_sf_reward,
                                                                               False,
                                                                               self.off_policy_count,
                                                                               self.exploration_policy_count)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train_expl')
                    #print("if not seed_until_step(self.global_step)_end ")
                ######################################################################

            # take env step
            #print("take env step(action) : ", action)
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    ##############################################################
    def save_explor_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_explor_{self.global_frame}.pt'
        keys_to_save = ['agent_exploration', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    ##############################################################

@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
