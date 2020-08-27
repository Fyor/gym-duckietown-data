import argparse
import os
from collections import defaultdict

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils import DrawLine
from gym_duckietown.envs import DuckietownEnv, DuckietownForward, DuckietownRewardFunc
import shutil

maplist = "4way  loop_dyn_duckiebots  loop_empty  loop_obstacles  loop_pedestrians  regress_4way_adam  regress_4way_drivable  small_loop_cw  small_loop  straight_road  straight_turn  udem1  zigzag_dists"

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0',
                                 epilog="Available maps:\n" + maplist)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--L2', type=float, default=0, help='L2 regularisation factor (1e-3 way too high, think 1e-6)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N',
                    help='repeat action in N frames (default: 4 used to be 8)')
parser.add_argument('--episode-length', type=int, default=500, metavar='N',
                    help='maximum steps per episode (old default: 1000)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--buffer-size', type=int, default=2000, help='replay buffer capacity (default: 2000)')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--fixed-speed', type=float, default=.1, help='Fixed velocity, 0 for disable (default: .1)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--checkpoint', action='store_true', help='Continue from last model')
parser.add_argument('--vis', action='store_true', help='plot with visdom')
parser.add_argument("--force", action="store_true", help="overwrite name")
parser.add_argument("--domain-rand", action="store_true", help="Enable domain randomisation")
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
# parser.add_argument('--maps', nargs='+', help='Maps to use', default=["loop_empty"])
parser.add_argument('--maps', nargs='+', help='Maps to use', default=["zigzag_dists"])

if __name__ == '__main__':
    parser.add_argument("name", type=str, help="Experiment name")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def get_tensorboard(args):
    logdir_base = "logs"
    if args.name == "test":
        logdir_base = "/tmp/tensorboard"

    logdir = logdir_base + "/" + args.name + "/"

    if args.force:
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError as e:
            pass

    if not os.path.exists(logdir_base):
        os.mkdir(logdir_base)

    prev = set(os.listdir(logdir_base))
    if args.name in prev:
        raise NameError("Experiment already exists")

    writer = SummaryWriter(logdir)
    writer.add_text("Info", str(vars(args)))
    return writer


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


DUCKIETOWN = True

action_shape = (2,) if DUCKIETOWN else (3,)
transition = np.dtype(
    [('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, action_shape), ('a_logp', np.float64),
     ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        # self.env = gym.make('CarRacing-v0')
        env = DuckietownEnv(
            seed=args.seed,  # random seed
            map_name=args.maps[0],
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=args.domain_rand,
            camera_width=96,
            camera_height=96,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            randomize_maps_on_reset=len(args.maps) > 1,
            camera_FOV_y=108
        )

        args.__setattr__("Env", type(env))

        # limit maps
        env.map_names = args.maps
        env.reset()

        assert (type(env) == DuckietownEnv or issubclass(type(env), DuckietownEnv)) == DUCKIETOWN
        env = DtRewardWrapper(env)
        self.env = env

        # self.reward_threshold = self.env.spec.reward_threshold
        self.reward_threshold = 100_000

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, done, info = self.env.step(action)
            if not DUCKIETOWN:
                # don't penalize "die state"
                if done:
                    reward += 100
                # green penalty
                if np.mean(img_rgb[:, :, 1]) > 185.0:
                    reward -= 0.05
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
        assert len(self.stack) == args.img_stack
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


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        out = action_shape[0]
        self.alpha_head = nn.Sequential(nn.Linear(100, out), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, out), nn.Softplus())
        self.apply(self._weights_init)

        self.i = 0
        self.latent_spaces = np.zeros((5, 256))

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)

        # self.latent_spaces[self.i % len(self.latent_spaces)] = x
        # self.i+= 1
        # if self.i>5:
        #     variance = np.var(self.latent_spaces, axis=0)
        #     variance_sum = np.sum(variance)
        #     print("Variance sum", variance_sum)

        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = args.buffer_size, args.batch_size  # 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr,
                                    weight_decay=args.L2, # L2 weight_decay=1e-5 and from paper 5e-3
                                    )
        self.best_reward = -np.inf

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        if torch.isnan(torch.sum(action)):
            raise ValueError("Predicion includes NaN")

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, last_reward):
        if last_reward > self.best_reward:
            self.best_reward = last_reward
            torch.save(self.net.state_dict(), f'param/ppo_net_params_{args.name}.pkl')

    def load_param(self):
        # self.net.load_state_dict(torch.load(f'param/ppo_net_params_base.pkl'))
        self.net.load_state_dict(torch.load(f'param/checkpoint.pkl'))

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # Advantage normalisation (Is this reward scaling?)
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient norm clipping
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


class DeathLogger:
    def __init__(self, writer):
        self.death_counter = defaultdict(int, {"invalid-pose": 0, "max-steps-reached": 0})
        self.writer = writer

    def log(self, info, i_ep):
        ### done_code options
        # invalid-pose
        # max-steps-reached
        # in-progress
        # stuck

        reason = info['Simulator']['done_code']
        if reason == "in-progress":
            raise ValueError("Simulator is done but gave no reason", info)
        self.death_counter[reason] += 1

        if i_ep % args.log_interval == 0:
            for cause, count in self.death_counter.items():
                writer.add_scalar(f"Deaths/{cause}", count / args.log_interval, i_ep)
                self.death_counter[cause] = 0


if __name__ == "__main__":
    # Next training changes:
    # Gradient norm clipping

    # todo https://www.reddit.com/r/reinforcementlearning/comments/7s8px9/deep_reinforcement_learning_practical_tips/
    # weight clipping
    # logvalue.clamp(-np.log(1e-5), np.log(1e-5))
    # Advantage normalisation
    agent = Agent()

    if args.checkpoint:
        print("Using pretrained weights")
        agent.load_param()

    env = Env()
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO" + args.name, xlabel="Episode",
                               ylabel="Moving averaged episode reward")

    writer = get_tensorboard(args)
    death_logger = DeathLogger(writer=writer)

    training_records = []
    running_score = 0
    reward_dist = np.zeros((1, args.log_interval))
    checkpoint_counter = np.zeros((1, args.log_interval))
    timesteps_counter = np.zeros((1, args.log_interval))
    state = env.reset()

    for i_ep in range(100000):
        score = 0
        state = env.reset()

        t = 0
        reward = 0
        action_history = np.zeros((1, args.episode_length, 2))
        for t in range(args.episode_length):
            action, a_logp = agent.select_action(state)
            if DUCKIETOWN:
                if args.fixed_speed != 0:
                    action[0] = args.fixed_speed  # Fixed speed
                # action[1] += np.random.normal(0, 0.1)
                # action = action.clip(0,1)
                # action_history[0,t] = action
                state_, reward, done, info = env.step(action * np.array([1., 2.]) + np.array([0, -1]))
            else:
                state_, reward, done, info = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
                # print(F"Reward {reward:7.1f}", "Death", info['Simulator']['done_code'])

            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done:
                break
        else:
            info['Simulator']['done_code'] = "max-steps-reached"

        running_score = running_score * 0.99 + score * 0.01
        reward_dist[0, i_ep % args.log_interval] = reward
        if isinstance(env.env.unwrapped, DuckietownRewardFunc):
            checkpoint_counter[0, i_ep % args.log_interval] = env.env.unwrapped.checkpoint_counter
        timesteps_counter[0, i_ep % args.log_interval] = t
        death_logger.log(info, i_ep)

        if i_ep % args.log_interval == 0:
            if args.vis:
                draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))

            writer.add_scalar('Reward/reward', running_score, i_ep)

            writer.add_histogram('Timesteps/timesteps', timesteps_counter, i_ep)
            if isinstance(env.env.unwrapped, DuckietownRewardFunc):
                writer.add_histogram('Reward/checkpoints', checkpoint_counter, i_ep)
            writer.add_histogram('Reward/reward_dist', reward_dist, i_ep)
            # only add actions which arent zero, in case
            # dirs = action_history[:, 1]
            # writer.add_histogram('Actions/last_actions_dist_vel', dirs[dirs != 0], i_ep)

            agent.save_param(running_score)
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
