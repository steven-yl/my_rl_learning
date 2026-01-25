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
    parser.add_argument("--rl_agent_type", type=str, default="Target_Net", help="RL agent type")
    args = parser.parse_args()

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    rl_agent_type = args.rl_agent_type
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "runs/DQN_{}/{}/{}".format(rl_agent_type, env_name, date)
    pth_name = "DQN_{}.pth".format(rl_agent_type)

    os.makedirs(output_dir, exist_ok=True)
    tb_writer = tb.SummaryWriter(output_dir)

    USE_PTH = True
    SAVE_PTH = True
    n_episodes = 30000
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    print("N_F: {}, N_A: {}".format(N_F, N_A))
    if rl_agent_type == "Naive":
        agent = models.NaiveDQN(N_F, N_A,\
                           lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1 )
    elif rl_agent_type == "ExperienceReplay":
        agent = models.ExperienceReplayDQN(N_F, N_A,\
                        lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1, capacity=20000, batch_size=32 )
    elif rl_agent_type == "Target_Net":
        agent = models.TargetNetDQN(N_F, N_A,\
                        lr=0.01, gamma=0.9, epsilon=1.0, eps_dec=1e-5, eps_min=1e-1, capacity=20000, batch_size=32, target_replace_freq=500 )
    else:
        raise ValueError("rl_agent_type must be Naive, ExperienceReplay or Target_Net")
    tb_writer.add_graph(agent, torch.randn((1, N_F)))

    if USE_PTH and os.path.exists(os.path.join(output_dir, pth_name)):
        agent.load_state_dict(torch.load(os.path.join(output_dir, pth_name), weights_only=False))
        print("Load DQN_{}.pth successfully".format(rl_agent_type))
    print("agent.gamma: {}".format(agent.gamma))
    agent.train()
    print("Start training...")

    def save_model_on_exit(signum, frame):
        torch.save(agent.state_dict(), os.path.join(output_dir, "DQN_{}.pth".format(rl_agent_type)))
        print(f"\nModel saved to {os.path.join(output_dir, 'DQN_{}.pth'.format(rl_agent_type))} due to abnormal termination (signal {signum}).")
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
            result = agent.learn(torch.FloatTensor(state), torch.LongTensor([action]),\
                               torch.FloatTensor([reward]), torch.FloatTensor(next_state))
            loss_ep += result["loss"]
            step += 1
            state = next_state
        if i % 1000 == 0:
            if SAVE_PTH:
                torch.save(agent.state_dict(), os.path.join(output_dir, "DQN_{}_{}.pth".format(rl_agent_type, i)))
                print("Save DQN_{}_{}.pth successfully at episode {}".format(rl_agent_type, i, i))
        tb_writer.add_scalar("DQN_{}/i_episode".format(rl_agent_type), i, i)
        tb_writer.add_scalar("DQN_{}/score".format(rl_agent_type), score, i)
        tb_writer.add_scalar("DQN_{}/loss".format(rl_agent_type), loss_ep/step if step > 0 else 0.0, i)
        tb_writer.add_scalar("DQN_{}/agent_epsilon".format(rl_agent_type), agent.epsilon, i)
        tb_writer.add_scalar("DQN_{}/q_pred".format(rl_agent_type), result["q_pred"], i)
        tb_writer.add_scalar("DQN_{}/q_target".format(rl_agent_type), result["q_target"], i)
        tb_writer.add_scalar("DQN_{}/is_update_target_net".format(rl_agent_type), result["is_update_target_net"], i)
    if SAVE_PTH:
        torch.save(agent.state_dict(), os.path.join(output_dir, "DQN_{}_{}.pth".format(rl_agent_type, i)))
    # close tensorboard writer
    tb_writer.close()
