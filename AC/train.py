import gymnasium as gym
import models
import torch.utils.tensorboard as tb
import os
import torch
import datetime

def generate_env(env_name: str, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    return {"env": env, "N_F": N_F, "N_A": N_A}

def main():
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = tb.SummaryWriter("runs/AC/CartPole-v1/" + date)
    USE_PTH = False
    SAVE_PTH = False
    MAX_EPISODES = 800
    MAX_EP_STEPS = 2000
    RENDER = False
    GAMMA = 0.6
    LR_A = 0.001
    LR_C = 0.01
    env_property = generate_env("CartPole-v1", render_mode="human" if RENDER else None)
    env = env_property["env"]
    actor = models.Actor(env_property["N_F"], env_property["N_A"], lr = LR_A)
    critic = models.Critic(env_property["N_F"], lr = LR_C, GAMMA = GAMMA)
    if USE_PTH and os.path.exists("./AC/actor.pth"):
        actor.load_state_dict(torch.load("./AC/actor.pth", weights_only=False))
    if USE_PTH and os.path.exists("./AC/critic.pth"):
        critic.load_state_dict(torch.load("./AC/critic.pth", weights_only=False))
    for i_episode in range(MAX_EPISODES):
        s, info = env.reset(seed=1)
        t = 0
        episode_reward = 0
        while True:   # s, a, r, s_
            a = actor.choose_action(s)
            s_, r, terminated, truncated, info = env.step(a)
            done = terminated
            if done:
                r = -20
            td_error = critic.learn(s, r, s_)
            actor.learn(td_error)
            s = s_
            episode_reward += r
            t += 1
            if done or t >= MAX_EP_STEPS:
                if 'episode_reward' not in globals():
                    running_reward = episode_reward
                else:
                    running_reward = running_reward * 0.95 + episode_reward * 0.05
                # 训练时通常不渲染，以提高训练速度
                # 如需可视化，请使用 ac_sim.py 加载训练好的模型
                # print("Episode: {} - Reward: {} - Running Reward: {}".format(i_episode, episode_reward, running_reward))
                tb_writer.add_scalar("AC/i_episode", i_episode, global_step=i_episode)
                tb_writer.add_scalar("AC/Reward", episode_reward, global_step=i_episode)
                tb_writer.add_scalar("AC/Running Reward", running_reward, global_step=i_episode)
                # INSERT_YOUR_CODE
                # 可视化模型权重参数分布到TensorBoard
                for name, param in actor.named_parameters():
                    tb_writer.add_histogram(f"AC/actor/{name}", param, global_step=i_episode)
                for name, param in critic.named_parameters():
                    tb_writer.add_histogram(f"AC/critic/{name}", param, global_step=i_episode)
                break
        # 保存模型
        if SAVE_PTH:
            torch.save(actor.state_dict(), "./AC/actor.pth")
            torch.save(critic.state_dict(), "./AC/critic.pth")
        tb_writer.close()
    env.close()

if __name__ == "__main__":
    main()