import argparse
import time

import numpy as np

# import duckie

import gym
import torch
import torch.nn as nn
from tqdm import trange

from train import DuckietownHistoryEnvNormal, Net

import gym_duckietown

from utils import velangle_to_lrpower

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

    def load_param(self, file):
        self.net.load_state_dict(torch.load(file, map_location=device))


if __name__ == "__main__":
    table = []
    for agent_weights in ['param/checkpoint.pkl', ]:
        # for map_name in ["zigzag_dists", "straight_road", "udem1", "small_loop", "loop_empty"]:
        for map_name in ["duckies-rand"]:


            agent = Agent()
            print("Loading weigths", agent_weights)
            agent.load_param(file=agent_weights)

            seed = np.random.randint(0, 100)
            print("Seed", seed)
            env = DuckietownHistoryEnvNormal(maps=[map_name], action_repeat=8, seed=seed)

            training_records = []
            running_score = 0
            state = env.reset()
            lane_dist = []
            deaths = 0
            deaths_after_20 = 0
            good = 0
            ts = []
            speed = []

            for i_ep in trange(500):
                score = 0
                state = env.reset()

                try:
                    for t in range(500):
                        action = agent.select_action(state)
                        action = action * 2 - 1

                        # action[0] = max(action[0], .1)  # fixed velocity
                        action[0] = max(.5 - np.abs(action[1]), .1)  # Speed up when going straight

                        action = velangle_to_lrpower(action)
                        state_, reward, done, info = env.step(action)

                        # get lane distance
                        sim = env.env
                        lp = sim.get_lane_pos2(sim.cur_pos, sim.cur_angle)
                        d = np.abs(lp.dist)
                        lane_dist.append(d)

                        speed.append(sim.speed)

                        if args.render:
                            env.render()
                            # time.sleep(1/40)

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

                print(env.env.map_name, f'Ep {i_ep}\tScore: {score:.2f}\tLen:{t}')

            d = lane_dist

            print("Map", map_name, agent_weights)
            print(
                f"DATA: map deaths mean var & {map_name} & {deaths_after_20} & {(np.mean(d) * 100) / 0.585:.1f} \\%  & {(np.std(d) * 100) / 0.585:.1f} \\% \\ \\    Tile size 0.585")
            table.append(
                f" {agent_weights} & {np.mean(speed):.4f}m/s avg & {map_name} & {deaths_after_20}/100 & {(np.mean(d) * 100) / 0.585:.1f} \\%  & {(np.std(d) * 100) / 0.585:.1f} \\% \\\\ ")
            print(f" (Total deaths including sub20 steps {deaths}) mean steps before death {np.mean(ts)}")
            print()

    print("TABLE")
    print("\n".join(table))
