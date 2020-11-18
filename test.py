import argparse
import time

import numpy as np

import gym
import torch
import torch.nn as nn
from tqdm import trange

from train import Env, Net

import gym_duckietown

from train_ray import velangle_to_lrpower

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, file=None):
        # self.net.load_state_dict(torch.load('param/checkpoint.pkl', map_location=device))
        # self.net.load_state_dict(torch.load('param/ppo_net_params_zigzag_fast_0.25.pkl', map_location=device)) # 0.3 speed
        file = file or 'param/ppo_net_params_check_straight_no_l2.pkl'
        self.net.load_state_dict(torch.load(file, map_location=device))
DUCKIETOWN = True

# "loop_empty  small_loop_cw  small_loop  straight_road   udem1  zigzag_dists"
if __name__ == "__main__":
    table = []
    # for agent_weights, agent_speed in [('param/ppo_net_params_zigzag_fast_0.25.pkl', 0.25)]: #[ ('param/checkpoint.pkl', 0.05)]: # ('param/ppo_net_params_check_straight_no_l2.pkl', 0.2), ('param/ppo_net_params_zigzag_fast_0.25.pkl', 0.3),
    for agent_weights, agent_speed in [('param/checkpoint.pkl', 0.25)]: #[ ('param/checkpoint.pkl', 0.05)]: # ('param/ppo_net_params_check_straight_no_l2.pkl', 0.2), ('param/ppo_net_params_zigzag_fast_0.25.pkl', 0.3),
        for map_name in ["udem1", "zigzag_dists", "small_loop", "loop_empty"]:


            agent = Agent()
            agent.load_param(file=agent_weights)
            # env = Env(maps=["zigzag_dists", "small_loop", "loop_empty"], action_repeat=8)
            seed = np.random.randint(0,100)
            print("seed", seed)
            env = Env(maps=[map_name], action_repeat=2, seed=seed)


            training_records = []
            running_score = 0
            state = env.reset()
            lane_dist = []
            deaths = 0
            deaths_after_20 = 0
            good = 0
            ts = []

            for i_ep in trange(500):
                score = 0
                state = env.reset()

                try:
                    for t in range(500):
                        action = agent.select_action(state)
                        if DUCKIETOWN:
                            # veldir
                            action[0] = agent_speed  # 0.3 or 0.1 # Fixed speed
                            action = action * np.array([1., 2.]) + np.array([0., -1.])
                            # lr power
                            action = velangle_to_lrpower(action)
                            # exec
                            state_, reward, done, info = env.step(action )

                            sim = env.env
                            lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
                            d = np.abs(lp.dist)
                            lane_dist.append(d)

                        else:
                            state_, reward, done, info = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                        if args.render:
                            env.render()
                            time.sleep(1/40)
                        score += reward
                        state = state_
                        if done:
                            deaths += 1
                            if t > 10:
                                deaths_after_20 += 1
                                good += 1
                                ts.append(t)
                            break
                    else:
                        good += 1
                        ts.append(t)

                    if good > 100:
                        break
                except gym_duckietown.simulator.NotInLane as e:
                    print(e)

                # print(env.env.map_name, f'Ep {i_ep}\tScore: {score:.2f}\tLen:{t}')

            d = lane_dist

            print("Map", map_name, agent_weights, agent_speed)
            print(f"DATA: map deaths mean var & {map_name} & {deaths_after_20} & {(np.mean(d) * 100) / 0.585:.1f} \\%  & {(np.std(d)* 100) / 0.585:.1f} \\% \\ \\    Tile size 0.585")
            table.append(f" {agent_weights} & {agent_speed*1.4:.4f}m/s & {map_name} & {deaths_after_20}/100 & {(np.mean(d) * 100) / 0.585:.1f} \\%  & {(np.std(d)* 100) / 0.585:.1f} \\% \\\\ ")
            print(f" (Total deaths including sub20 steps {deaths}) mean steps before death {np.mean(ts)}")
            print()

    print("TABLE")
    print("\n".join(table))