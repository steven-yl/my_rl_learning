import torch
import models
import torch.utils.tensorboard as tb
import gymnasium as gym
import os
import datetime
import argparse
import signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--rl_agent_type", type=str, default="Naive", help="RL agent type")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")

    rl_agent_type = args.rl_agent_type
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = tb.SummaryWriter("runs/DQN_{}/LunarLander-v3/{}".format(rl_agent_type, date))

    USE_PTH = True
    SAVE_PTH = True
    n_episodes = 30000
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    print("N_F: {}, N_A: {}".format(N_F, N_A))
    if rl_agent_type == "Naive":
        agent = models.DQN(N_F, N_A,\
                           lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1 )
    elif rl_agent_type == "StoreTransition":
        agent = models.DQN_StoreTransition(N_F, N_A,\
                        lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1, capacity=20000, batch_size=32 )
    elif rl_agent_type == "Double_Net":
        agent = models.DQN_Double_Net(N_F, N_A,\
                        lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1, capacity=20000, batch_size=32 )
    else:
        raise ValueError("rl_agent_type must be Naive, StoreTransition or Double_Net")
    tb_writer.add_graph(agent, torch.randn((1, N_F)))

    if USE_PTH and os.path.exists("./DQN/DQN_{}.pth".format(rl_agent_type)):
        agent.load_state_dict(torch.load("./DQN/DQN_{}.pth".format(rl_agent_type), weights_only=False))
        print("Load DQN_{}.pth successfully".format(rl_agent_type))
    print("agent.gamma: {}".format(agent.gamma))
    agent.train()
    print("Start training...")

    def save_model_on_exit(signum, frame):
        torch.save(agent.state_dict(), "./DQN/DQN_{}.pth".format(rl_agent_type))
        print(f"\nModel saved to ./DQN/DQN_{rl_agent_type}.pth due to abnormal termination (signal {signum}).")
        tb_writer.close()
        exit(0)

    # Register the handler for SIGINT (Ctrl+C), SIGTERM (kill), etc.
    signal.signal(signal.SIGINT, save_model_on_exit)
    signal.signal(signal.SIGTERM, save_model_on_exit)

    for i in range(n_episodes):
        score = 0
        done = False
        state, info = env.reset(seed=i)
        loss_ep = 0
        step = 0
        while not done:
            action = agent.choose_action(torch.FloatTensor(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            loss = agent.learn(torch.FloatTensor(state), torch.LongTensor([action]),\
                               torch.FloatTensor([reward]), torch.FloatTensor(next_state))
            loss_ep += loss
            step += 1
            state = next_state
        if i % 1000 == 0:
            if SAVE_PTH:
                torch.save(agent.state_dict(), "./DQN/DQN_{}.pth".format(rl_agent_type))
                print("Save DQN_{}.pth successfully at episode {}".format(rl_agent_type, i))
        tb_writer.add_scalar("DQN_{}/i_episode".format(rl_agent_type), i, i)
        tb_writer.add_scalar("DQN_{}/score".format(rl_agent_type), score, i)
        tb_writer.add_scalar("DQN_{}/loss".format(rl_agent_type), loss_ep/step, i)
        tb_writer.add_scalar("DQN_{}/agent_epsilon".format(rl_agent_type), agent.epsilon, i)
    if SAVE_PTH:
        torch.save(agent.state_dict(), "./DQN/DQN_{}.pth".format(rl_agent_type))
    # close tensorboard writer
    tb_writer.close()
