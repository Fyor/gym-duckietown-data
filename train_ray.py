import os

import cv2
import ray
from gym_duckietown.simulator import Simulator
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from gym_duckietown.envs import DuckietownEnv

from ray_model import VisionNetwork
import numpy as np
import gym



class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward < -10:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


class DuckieTownHistoryEnv(gym.Env):
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, ray_config):
        maps = ["zigzag_dists"]
        self.action_repeat = 8
        seed = np.random.randint(0, 1000)
        domain_rand = False
        self.img_stack = 4

        # self.env = gym.make('CarRacing-v0')
        self.env = DuckietownEnv(
            seed=seed,  # random seed
            map_name=maps[0],
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=domain_rand,
            # camera_width=96, DONE IN GRAY TO FIX BUG
            # camera_height=96,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            randomize_maps_on_reset=len(maps) > 1,
            # camera_FOV_y=108
            distortion=True,
        )

        self.env = DtRewardWrapper(self.env)

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(4, 96, 96), dtype=np.float32
        )

        # limit maps
        self.env.map_names = maps
        self.env.reset()

        # self.env = DtRewardWrapper(self.env)

        # self.reward_threshold = self.env.spec.reward_threshold
        self.reward_threshold = 100_000

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):

        action = np.array(action)
        action[0] = .1 # SET FIXED VELOCITY TODO TEST

        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, done, info = self.env.step(action)
            total_reward += reward
            # if no reward recently, end the episode
            # if self.av_r(reward) <= -0.1:
            #     done = True
            #     info['Simulator']['msg'] += "Too little reward"

            if done:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        if os.getlogin() == "thomas" and __name__ == "__main__":
            self.env.render()
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, info

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):

        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        gray = cv2.resize(gray, dsize=(96, 96))
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

if __name__ == '__main__' and os.getlogin() == "thomas":
    ray.init()

    agent = PPOTrainer(config={"env": DuckieTownHistoryEnv,
                               "framework": "torch",
                               "model": {
                                   # Extra kwargs to be passed to your model's c'tor.
                                   # "custom_model_config": {
                                   #     "fcnet_hiddens": [512, 512, 512, 256, ],
                                   #     "fcnet_activation": "relu",
                                   # },
                                   "custom_model": VisionNetwork,
                               },
                               "num_workers": 3,
                               "num_gpus": .1,
                               "lr": 1e-3,  # tune.grid_search([1e-2, 1e-4, 1e-6]),
                               "train_batch_size": 2000,
                               "rollout_fragment_length": 400,
                               "sgd_minibatch_size": 128,  # tune.grid_search([32, 64, 128])
                               "num_sgd_iter": 10,
                               "clip_param": 0.1,
                               "grad_clip": .5,
                               }, )

    i = 9
    agent.restore(f"param_ray/checkpoint_{i}0/checkpoint-{i}0")
    agent.train()


if __name__ == '__main__':

    ray.init()
    result = tune.run(PPOTrainer,  # PPOTrainer
                      config={"env": DuckieTownHistoryEnv,
                              "framework": "torch",
                              "model": {
                                  "custom_model": VisionNetwork,
                              },
                              "num_workers": 3,
                              "num_gpus": .1,
                              "lr": 1e-3,  # tune.grid_search([1e-2, 1e-4, 1e-6]),
                              "train_batch_size": 2000,
                              "rollout_fragment_length": 400,
                              "sgd_minibatch_size": 128,
                              "num_sgd_iter": 10,
                              "clip_param": 0.1,
                              "grad_clip": .5,
                              },
                      stop={
                          # "training_iteration": 50000,  # episodes or steps?
                          "time_total_s": 60 * 60 * 8  # 10 hours
                      },
                      # resources_per_trial={"cpu": 1}
                      num_samples=3,
                      checkpoint_freq=50,  # iterations
                      checkpoint_at_end=True,
                      )


