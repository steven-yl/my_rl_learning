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
    env_property = generate_env("CartPole-v1", render_mode="human")
    env = env_property["env"]
    actor = models.Actor(env_property["N_F"], env_property["N_A"], lr = LR_A)
    if os.path.exists("./AC/actor.pth"):
        actor.load_state_dict(torch.load("./AC/actor.pth", weights_only=False))
        print("actor model loaded")

    actor.eval()
    for i_episode in range(MAX_EPISODES):
        s, info = env.reset(seed=1)
        while True:
            a = actor.choose_action(s)
            s_, r, terminated, truncated, info = env.step(a)
            done = terminated
            s = s_
            if done:
                break
    env.close()

if __name__ == "__main__":
    main()