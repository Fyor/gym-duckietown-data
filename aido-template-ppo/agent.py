import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

img_stack = 4
action_shape = (2,)
buffer_size = 2000
batch_size = 128
lr = 1e-3
L2 = 0
gamma = 0.99
seed = np.random.randint(0, 1000)
print("SEED", seed)


transition = np.dtype(
    [('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, action_shape), ('a_logp', np.float64),
     ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)



class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()



        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
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

transition = np.dtype(
    [('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, action_shape), ('a_logp', np.float64),
     ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])



class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = buffer_size, batch_size  # 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().float().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr,
                                    weight_decay=L2, # L2 weight_decay=1e-5 and from paper 5e-3
                                    )
        self.best_reward = -np.inf

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
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
            torch.save(self.net.state_dict(), f'param/ppo_net_params_save.pkl')

    def load_param(self, filename=f'param/checkpoint.pkl'):
        self.net.load_state_dict(torch.load(filename, map_location=device))

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

        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.float).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
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
