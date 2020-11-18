import time

import ray
import cv2
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from gym_duckietown.envs import DuckietownEnv

from ray_model import VisionNetwork
import numpy as np
import gym

# Restored previous trial from the given checkpoint
# from test import velangle_to_lrpower
from train_ray import DuckieTownHistoryEnv, velangle_to_lrpower
from utils import Display

ray.init()

agent = PPOTrainer(config={"env": DuckieTownHistoryEnv,
              "framework": "torch",
              "model": {
                  "custom_model": VisionNetwork,
              },
            "num_workers": 0,
            "num_gpus": 0,
            })
i = 9
agent.restore(f"param_ray/checkpoint_{i}0/checkpoint-{i}0")


env = DuckieTownHistoryEnv({})
# display = Display()

for i_ep in range(500):
    score = 0
    state = env.reset()
    treward = 0
    for t in range(500):
        # state = np.zeros_like(state, dtype=np.float32)
        # if state.shape != (4, 96, 96):
        #     state = np.array([cv2.resize(channel, dsize=(96, 96)) for channel in state])
        # # print("Shape", state.shape, state[0].max())
        # out = cv2.resize(((state[0]+1)*128).astype(np.uint8), dsize=(96*4, 96*4))
        # cv2.imshow("a", out)
        # cv2.waitKey(1)
        # display.show(state[0])

        action = agent.compute_action(state)
        print(action)
        # veldir
        # action[0] = .3  # 0.3 or 0.1 # Fixed speed
        # action = action * np.array([1., 2.]) + np.array([0., -1.])
        # lr power
        # action = velangle_to_lrpower(action)
        # exec
        state, reward, done, info = env.step(action)
        treward += reward
        env.render()
        time.sleep(1 / 40)

        if done:
            break
    print("Reward:", reward)

# display.close()