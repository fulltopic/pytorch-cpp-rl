"""
Adapted from:
github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py

Provides utility functions for making Gym environments.
"""
import gym
from gym.spaces import Box
import numpy as np

import logging

from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import (VecNormalize
                                                    as VecNormalize_)


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # print(obs.sum())
        # print(obs)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=0)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        # print("----------> step_wait")
        # print(self.stackedobs.shape)
        # print(obs.shape)
        return self.stackedobs, rews, news, infos

    def reset(self):
        # logging.info("VecFrameStack reset")
        print('------------------------> vecframestack reset')
        obs = self.venv.reset()
        # logging.info("Total obs shape %ss", obs.shape)
        # logging.info("Get reset result: %d", obs.size)
        self.stackedobs[...] = 0
        self.stackedobs[-obs.shape[-1]:, ...] = obs
        # logging.info("stackedobs: %d, %ss", self.stackedobs.size, self.stackedobs.shape)
        return self.stackedobs

    #TODO zf: index and last frame update
    def resetOne(self, x):
        # logging.info("VecFrameStack resetOne %d", x)
        obs = self.venv.resetOne(x)
        # logging.info("stack shape %ss", self.stackedobs.shape)
        # logging.info("obs shape %ss", obs.shape)
        self.stackedobs[x, ...] = 0
        self.stackedobs[x, self.stackedobs.shape[1] - 1, ...] = obs
        # self.stackedobs[...] = 0
        # self.stackedobs[-obs.shape[-1]:, ...] = obs
        return self.stackedobs

# class VecFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack):
#         self.venv = venv
#         self.nstack = nstack
#         wos = venv.observation_space  # wrapped ob space
#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)
#         self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)
#
#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         last_ax_size = obs.shape[-1]
#         # print(infos)
#         self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
#         for i, done in enumerate(news):
#             if done:
#                 if 'terminal_observation' in infos[i]:
#                     old_terminal = infos[i]['terminal_observation']
#                     new_terminal = np.concatenate(
#                         (self.stackedobs[i, ..., :-last_ax_size], old_terminal), axis=-1)
#                     infos[i]['terminal_observation'] = new_terminal
#                     print("info[i]: ")
#                     print(infos[i])
#                 else:
#                     print(
#                         "VecFrameStack wrapping a VecEnv without terminal_observation info")
#                 self.stackedobs[i] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs, rews, news, infos
#
#     def reset(self):
#         # logging.info("VecFrameStack reset")
#         obs = self.venv.reset()
#         # logging.info("Total obs shape %ss", obs.shape)
#         # logging.info("Get reset result: %d", obs.size)
#         self.stackedobs[...] = 0
#         self.stackedobs[-obs.shape[-1]:, ...] = obs
#         # logging.info("stackedobs: %d, %ss", self.stackedobs.size, self.stackedobs.shape)
#         return self.stackedobs
#
#     #TODO zf: index and last frame update
#     def resetOne(self, x):
#         # logging.info("VecFrameStack resetOne %d", x)
#         obs = self.venv.resetOne(x)
#         # logging.info("stack shape %ss", self.stackedobs.shape)
#         # logging.info("obs shape %ss", obs.shape)
#         self.stackedobs[x, ...] = 0
#         self.stackedobs[x, self.stackedobs.shape[1] - 1, ...] = obs
#         # self.stackedobs[...] = 0
#         # self.stackedobs[-obs.shape[-1]:, ...] = obs
#         return self.stackedobs


class VecRewardInfo(VecEnvWrapper):
    def __init__(self, venv):
        self.venv = venv
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # print(obs)
        infos = {'reward': np.expand_dims(rews, -1)}
        return obs, rews, news, infos

    def reset(self):
        logging.info("VecRewardInfo reset")
        obs = self.venv.reset()
        return obs

    def resetOne(self, x):
        # logging.info("VecRewardInfo resetOne %d", x)
        return self.venv.resetOne(x)
        # logging.info("RewardInfo obs shape %ss", obs.)
        # return obs


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        print("----------------------------------------->VecNormalize")
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean)
                          / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        infos = {'reward': np.expand_dims(rews, -1)}
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew,
                           self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos


# def make_env(env_id, seed, rank):
#     def _thunk():
#         print("make env by gym.make")
#         env = gym.make(env_id)
#         print("env made")
#
#         is_atari = hasattr(gym.envs, 'atari') and isinstance(
#             env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
#         print("is_atari", is_atari)
#         if is_atari:
#             print("To create AtariWrapper")
#             env = AtariWrapper(env)
#             print("Created atari wrapper")
#
#         env.seed(seed + rank)
#
#         obs_shape = env.observation_space.shape
#         print("obs_shape", obs_shape)
#
#         if is_atari:
#             if len(env.observation_space.shape) == 3:
#                 print("obs shape len == 3")
#                 env = EpisodicLifeEnv(env)
#         elif len(env.observation_space.shape) == 3:
#             raise NotImplementedError("CNN models work only for atari,\n"
#                                       "please use a custom wrapper for a "
#                                       "custom pixel input env.\n See "
#                                       "wrap_deepmind for an example.")
#
#         # If the input has shape (W,H,3), wrap for PyTorch convolutions
#         obs_shape = env.observation_space.shape
#         if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
#             env = TransposeImage(env)
#
#         return env
#     return _thunk

#
# def make_vec_envs(env_name, seed, num_processes, num_frame_stack=None):
#     print("make_vec_envs")
#     print(num_processes)
#     envs = [make_env(env_name, seed, i) for i in range(num_processes)]
#
#     if len(envs) > 1:
#         print("SubprocVecEnv")
#         envs = SubprocVecEnv(envs)
#     else:
#         print("DummyVecEnv")
#         envs = DummyVecEnv(envs)
#
#     print("VecRewardInfo")
#     envs = VecRewardInfo(envs)
#
#     if num_frame_stack is not None:
#         print("VecFrameStack")
#         envs = VecFrameStack(envs, num_frame_stack)
#     elif len(envs.observation_space.shape) == 3:
#         print("VecFrameStack")
#         envs = VecFrameStack(envs, 4)
#
#     return envs

def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    # print(env.spec.id)
    assert 'NoFrameskip' in env.spec.id
    # env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)

    # if "FIRE" in env.unwrapped.get_action_meanings():
    #     print('-------------------------> With fire action')
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            print('-----------------------------> is atari')
            env = make_atari(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if is_atari:
            if len(env.observation_space.shape) == 3:
                print(env_id)
                # if 'Breakout' in env_id:
                #     print('Special maxop for breakout')
                #     env = AtariWrapper(env, clip_reward=False, noop_max=1)
                # else:
                #     print('Not breakout')
                #     env = AtariWrapper(env, clip_reward=False)
                # env = AtariWrapper(env)
                env = AtariWrapper(env, clip_reward=False)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError("CNN models work only for atari,\n"
                                      "please use a custom wrapper for a "
                                      "custom pixel input env.\n See "
                                      "wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env
    return _thunk


def make_vec_envs(env_name, seed, num_processes, num_frame_stack=None):
    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    if len(envs) > 1:
        #envs = SubprocVecEnv(envs)
        envs = DummyVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecRewardInfo(envs)

    if num_frame_stack is not None:
        envs = VecFrameStack(envs, num_frame_stack)
    elif len(envs.observation_space.shape) == 3:
        envs = VecFrameStack(envs, 4)

    print('-----------------__------ created envs ', envs.num_envs)
    return envs
