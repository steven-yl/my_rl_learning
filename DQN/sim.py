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
    env_property = generate_env("LunarLander-v3", render_mode="human")
    env = env_property["env"]
    N_F = env_property["N_F"]
    N_A = env_property["N_A"]

    # 修复models中可用的agent类型: Naive、ExperienceReplay、Target_Net
    rl_agent_type = "ExperienceReplay"
    if rl_agent_type == "Naive":
        agent = models.NaiveDQN(
            N_F, N_A,
            lr=0.01, gamma=0.9, epsilon=0.0, eps_dec=1e-5, eps_min=0.0
        )
    elif rl_agent_type == "ExperienceReplay":
        agent = models.ExperienceReplayDQN(
            N_F, N_A,
            lr=0.01, gamma=0.9, epsilon=0.0, eps_dec=1e-5, eps_min=0.0,
            capacity=20000, batch_size=32
        )
    elif rl_agent_type == "Target_Net":
        agent = models.TargetNetDQN(
            N_F, N_A,
            lr=0.01, gamma=0.9, epsilon=0.0, eps_dec=1e-5, eps_min=0.0,
            capacity=20000, batch_size=32
        )
    else:
        raise ValueError("rl_agent_type must be Naive, ExperienceReplay or Target_Net")
    # 加载权重
    pth_path = "./DQN/DQN_{}.pth".format(rl_agent_type)
    if os.path.exists(pth_path):
        agent.load_state_dict(torch.load(pth_path, weights_only=False))
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