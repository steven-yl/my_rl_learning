import gymnasium as gym
import models
import os
import torch

def generate_env(env_name: str, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    return {"env": env, "N_F": N_F, "N_A": N_A}

def main():
    MAX_EPISODES = 3000
    LR_A = 0.001
    env_property = generate_env("LunarLander-v3", render_mode="human")
    env = env_property["env"]
    N_F = env_property["N_F"]
    N_A = env_property["N_A"]

    rl_agent_type = "Double_Net"
    if rl_agent_type == "Naive":
        agent = models.DQN(N_F, N_A,\
                           lr=0.001, gamma=0.6, epsilon=0.0, eps_dec=1e-5, eps_min=1e-1 )
    elif rl_agent_type == "StoreTransition":
        agent = models.DQN_StoreTransition(N_F, N_A,\
                        lr=0.001, gamma=0.9, epsilon=0.0, eps_dec=1e-5, eps_min=1e-1, capacity=10000, batch_size=128 )
    elif rl_agent_type == "Double_Net":
        agent = models.DQN_Double_Net(N_F, N_A,\
                        lr=0.001, gamma=0.9, epsilon=0.0, eps_dec=1e-5, eps_min=1e-1, capacity=10000, batch_size=128 )
    else:
        raise ValueError("rl_agent_type must be Naive, StoreTransition or Double_Net")
    if os.path.exists("./DQN/DQN_{}.pth".format(rl_agent_type)):
        agent.load_state_dict(torch.load("./DQN/DQN_{}.pth".format(rl_agent_type), weights_only=False))
        print("agent model loaded")
    agent.eval()
    for i_episode in range(MAX_EPISODES):
        s, info = env.reset(seed=1)
        while True:
            a = agent.choose_action(torch.FloatTensor(s))
            s_, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s = s_
            if done:
                break
    env.close()

if __name__ == "__main__":
    main()