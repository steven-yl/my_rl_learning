import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self,n_features, n_actions, hidden_size: int = 256):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) 

class NaiveDQN(nn.Module):
    def __init__(self, n_features, n_actions, lr: float, gamma: float, epsilon: float, eps_dec: float, eps_min: float) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        print("n_features: {}, n_actions: {}, lr: {}, gamma: {}, epsilon: {}, eps_dec: {}, eps_min: {}".format(
            n_features, n_actions, lr, gamma, epsilon, eps_dec, eps_min))

        self.eval_net = Net(n_features, n_actions)

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.to(self.device)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        return self.eval_net(state)
    
    def choose_action(self, state: torch.Tensor) -> int:
        if np.random.random() > self.epsilon:
            state = state.to(self.device)
            actions = self.eval_net.forward(state)
            action = torch.argmax(actions, dim=-1).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        states = state.to(self.device)
        actions = action.to(self.device)
        rewards = reward.to(self.device)
        next_states = next_state.to(self.device)
   
        q_pred = self.eval_net.forward(states)
        q_pred = torch.gather(q_pred, dim=-1, index=actions)
        q_target = rewards +  self.gamma * self.eval_net.forward(next_states).max(dim=-1, keepdim=True)[0]
        loss = self.loss_fn(q_pred, q_target)

        loss.backward()
        self.optimizer.step()  
        self.decrement_epsilon()

        return loss.item()
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

class ExperienceReplayDQN(NaiveDQN):
    def __init__(self, n_features: int, n_actions: int, lr: float, gamma: float,\
                epsilon: float, eps_dec: float, eps_min: float, capacity: int = 2000, batch_size: int = 32) -> None:
        super().__init__(n_features, n_actions, lr, gamma, epsilon, eps_dec, eps_min)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def store_transition(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, next_state:torch.Tensor) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    def learn(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> float:
        self.store_transition(state, action, reward, next_state)
        states, actions, rewards, next_states = self.sample_transition()
        loss = super().learn(states, actions, rewards, next_states)
        return loss 

    def sample_transition(self) -> tuple:
        transitions = np.random.choice(len(self.memory), min(self.batch_size, len(self.memory)))
        state, action, reward, next_state = zip(*[self.memory[i] for i in transitions])
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        next_state = torch.stack(next_state)
        return state, action, reward, next_state


class Memory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, next_state:torch.Tensor) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        transitions = np.random.choice(len(self.memory), min(batch_size, len(self.memory)))
        state, action, reward, next_state = zip(*[self.memory[i] for i in transitions])
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        next_state = torch.stack(next_state)
        return state, action, reward, next_state

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for i, priority in zip(indices, priorities):
            self.memory[i][2] = priority
class PrioritizedExperienceReplayDQN(NaiveDQN):
    def __init__(self, n_features: int, n_actions: int, lr: float, gamma: float,\
                epsilon: float, eps_dec: float, eps_min: float, capacity: int = 2000,\
                batch_size: int = 32, prioritized: bool = True) -> None:
        super().__init__(n_features, n_actions, lr, gamma, epsilon, eps_dec, eps_min)
        self.capacity = capacity
        self.batch_size = batch_size
        self.prioritized = prioritized
        if self.prioritized:
            self.memory = Memory(capacity)
        else:
            self.memory = []
            self.position = 0

    def store_transition(self, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor, next_state:torch.Tensor) -> None:
        if self.prioritized:
            self.memory.store(state, action, reward, next_state)
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = (state, action, reward, next_state)
            self.position = (self.position + 1) % self.capacity

    def sample_transition(self) -> tuple:
        if self.prioritized:
            return self.memory.sample(self.batch_size)
        else:
            transitions = np.random.choice(len(self.memory), min(self.batch_size, len(self.memory)))
            state, action, reward, next_state = zip(*[self.memory[i] for i in transitions])
            state = torch.stack(state)
            action = torch.stack(action)
            reward = torch.stack(reward)
            next_state = torch.stack(next_state)
            return state, action, reward, next_state

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if self.prioritized:
            self.memory.update_priorities(indices, priorities)

    def learn(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> float:

        states, actions, rewards, next_states = self.sample_transition()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        loss = super().learn(states, actions, rewards, next_states)
        return loss.item()

class TargetNetDQN(ExperienceReplayDQN):
    def __init__(self, n_features: int, n_actions: int, lr: float, gamma: float,\
                epsilon: float, eps_dec: float, eps_min: float, capacity: int = 2000,\
                batch_size: int = 32, target_replace_freq: int = 100) -> None:
        super().__init__(n_features, n_actions, lr, gamma, epsilon, eps_dec, eps_min, capacity, batch_size)
        self.target_net = Net(n_features, n_actions)
        self.target_net.to(self.device)
        self.target_replace_freq = target_replace_freq
        self.learn_step_counter = 0

    def learn(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor) -> float:
        is_update_target_net = False
        if self.learn_step_counter == 0 or self.learn_step_counter % self.target_replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("target net updated")
            is_update_target_net = True
        self.learn_step_counter += 1
        self.store_transition(state, action, reward, next_state)
        states, actions, rewards, next_states = self.sample_transition()
        self.optimizer.zero_grad()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
   
        q_pred = self.eval_net.forward(states)
        q_pred = torch.gather(q_pred, dim=-1, index=actions)
        with torch.no_grad():
            q_next = self.target_net.forward(next_states)
        q_target = rewards +  self.gamma * q_next.max(dim=-1, keepdim=True)[0]
        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        self.decrement_epsilon()
        result = {
            "loss": loss.item(),
            "q_pred": q_pred.mean().item(),
            "q_target": q_target.mean().item(),
            "is_update_target_net": is_update_target_net,
        }
        return result

