# my_rl_learning

| No. |         method         | done(√\x)    |
|:---:| :--------------------  | :----:       |
|  1  |      \*Qlearning       | √            |
|  2  |         Sarsa          | √            |
|  3  |      SarsaLambda       | x            |
|  4  |         \*DQN          | √            |
|  5  |      \*DQNwithPER      | x            |
|  6  |       DuelingDQN       | x            |
|  7  |   \*Policy Gradient    | x            |
|  8  |      \*AC and A2C      | x            |
|  9  |          ACER          | x            |
| 10  |          A3C           | x            |
| 11  |  \*SAC (PER optional)  | x            |
| 12  |         \*DDPG         | x            |
| 13  | TD3 (PER,HER optional) | x            |
| 14  |          \*TRPO        | x            |
| 15  |         \*PPO          | x            |
| 16  |          DPPO          | x            |
| 17  | Multi-Agent DDPG       | x            |
| 18  |        TEMPPO          | ×            |

## 1. Qlearning
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$
 
 其中：
 - $Q(s_t, a_t)$：当前状态$s_t$下采取动作$a_t$的Q值
 - $\alpha$：学习率，控制Q值的更新幅度
 - $r_{t+1}$：在状态$s_t$下采取动作$a_t$后获得的即时奖励
 - $\gamma$：折扣因子，权衡未来奖励的影响
 - $\max_{a} Q(s_{t+1}, a)$：在新状态$s_{t+1}$下所有可能动作的最大Q值

## 2. Sarsa
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$
 
 其中：
  - $Q(s_t, a_t)$：当前状态$s_t$下采取动作$a_t$的Q值
  - $\alpha$：学习率，控制Q值的更新幅度
  - $r_{t+1}$：在状态$s_t$下采取动作$a_t$后获得的即时奖励
  - $\gamma$：折扣因子，权衡未来奖励的影响
  - $Q(s_{t+1}, a_{t+1})$：在新状态$s_{t+1}$下实际所选动作$a_{t+1}$的Q值
  
## 3. SarsaLambda
todo

## 4. DQN (Naive\ER\DoubleNet)
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$
 
 其中：
 - $Q(s_t, a_t)$：当前状态$s_t$下采取动作$a_t$的Q值
 - $\alpha$：学习率，控制Q值的更新幅度
 - $r_{t+1}$：在状态$s_t$下采取动作$a_t$后获得的即时奖励
 - $\gamma$：折扣因子，权衡未来奖励的影响
 - $\max_{a} Q(s_{t+1}, a)$：在新状态$s_{t+1}$下所有可能动作的最大Q值


#### 与Q-learning关键区别对比
| 特性 | Q-learning (表格法) | Naive DQN (Deep Q-Network) |
| :--- | :--- | :--- |
| **1. Q函数表示形式** | **表格**。为每个离散的 `(状态, 动作)` 对存储一个独立的 Q 值。 | **深度神经网络**。输入一个状态 `s`，输出所有可能动作 `a` 对应的 Q 值向量。 |
| **2. 状态处理能力** | 仅适用于**状态和动作空间离散且维度很低**的场景。状态维度稍高就会导致“维数灾难”，表格过大无法存储和学习。 | 可以处理**高维、连续的状态空间**（如图像、传感器数据）。神经网络具有强大的特征提取和泛化能力。 |
| **3. 核心挑战与解决方案** | 主要挑战是表格太大。没有专门针对“函数近似器”不稳定的解决方案。 | **挑战**：直接用神经网络替代表格会导致训练极其不稳定（数据相关性、非平稳目标）。<br>**两大支柱解决方案**：<br>• **经验回放**：将代理的经验 `(s, a, r, s’, done)` 存储到缓冲区，训练时随机采样一批数据。**打破数据间的时序相关性，提高数据效率**。<br>• **目标网络**：使用一个独立的、更新较慢的网络来计算 TD 目标 `Q_target`。**固定学习目标，缓解因目标不断变化带来的振荡和发散问题**。 |
| **4. 学习目标** | 直接更新表格中的数值。 | 最小化当前网络输出与目标网络计算出的 TD 目标之间的 **均方误差损失**。这是一个**监督学习回归问题**。 |
| **5. 适用场景** | 网格世界、简单游戏（如 Cliff Walking）等小规模离散问题。 | 复杂的视频游戏（如 Atari 2600）、机器人控制等高维感知和控制任务。 |


## 5. DQN(PER)
off-line算法一般要用到经验池，经验池随机抽样不能有效反复学习到有用的经验，所以有了PER(Prioritized Experience Replay)优先级经验回放。
- 如果算法足够好，少数有效经验也能学好，则没有必要用PER.
- 如果奖励函数足够dense，随机采用也能大概率采样到有效经验，则没有必要用PER.
- 使用PER速度会大大降低，性能不一定提高.
  - 需要维护优先级队列（如 SumTree 或堆结构），每次采样和更新都要操作这个数据结构（O(log n)）
  - 需要计算每个经验的 TD-error（时序差分误差）作为优先级，存储时多一步计算
  - 根据优先级加权采样，需要用 SumTree 等结构查找，然后还要计算采样概率的权重用于校正
  - 训练后需要重新计算被采样经验的 TD-error，并更新优先级，相当于每个 batch 多一次前向传播
  - PER 总是优先学习高 TD-error 的经验，可能过度关注这些样本，忽略其他重要经验，导致过拟合
- 如果CPU资源足够，奖励稀疏，可以采用PER.

## 6. DuelingDQN
todo

## 7. Policy Gradient
todo

## 8. AC and A2C
todo

## 9. ACER
todo

## 10. A3C
todo

## 11. SAC (PER optional)
todo

## 12. DDPG
todo

## 13. TD3 (PER,HER optional)
todo

## 14. TRPO
todo

## 15. PPO
todo

## 16. DPPO
todo

## 17. Multi-Agent DDPG
todo

## 18. TEMPPO
todo