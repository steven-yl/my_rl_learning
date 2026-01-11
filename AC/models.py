import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, n_features, n_actions, lr=0.001) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(-1)
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.action_prob_buffer = None

    def learn(self, td):
        log_probs = torch.log(self.action_prob_buffer)
        exp_v = log_probs * td.detach()
        loss = -exp_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return exp_v

    def choose_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).float()
        action_probs = self.net(state)
        action = np.random.choice(range(action_probs.shape[0]), p=action_probs.detach().numpy())
        self.action_prob_buffer = action_probs[action]
        return action


class Critic(nn.Module):
    def __init__(self, n_features, lr=0.001, GAMMA=0.9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.GAMMA = GAMMA

    def learn(self, s, r, s_):
        s = np.array(s, dtype=np.float32)
        s_ = np.array(s_, dtype=np.float32)
        s = torch.from_numpy(s).float()
        s_ = torch.from_numpy(s_).float()
        r = torch.tensor(r, dtype=torch.float32)
        with torch.no_grad():
            target_v = self.net(s_)
        pred_v = self.net(s)
        td_error = torch.mean((r + self.GAMMA * target_v) - pred_v)
        loss = td_error.square()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error