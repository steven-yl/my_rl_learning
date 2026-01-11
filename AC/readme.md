
## Actor-Critic算法核心公式

### Critic部分（状态价值估计，TD-Error）：

$$
TD\ error = \delta = r + \gamma V(s') - V(s)
$$

其中：

- $r$：即时奖励
- $\gamma$：折扣因子
- $V(s)$：当前状态$s$的价值估计
- $V(s')$：下一个状态$s'$的价值估计

Critic的目标是最小化TD error的平方损失：

$$
L_{critic} = (\delta)^2
$$

### Actor部分（策略更新）：

Actor用TD error来调整策略，目标是最大化：

$$
L_{actor} = - \log \pi(a | s) \cdot \delta
$$

其中：

- $\pi(a | s)$：当前策略下，给定状态$s$选择动作$a$的概率
- $\delta$：来自Critic的TD error

### 总结

- Critic评估策略的价值并计算TD error
- Actor根据TD error指引更新策略，使得动作选择概率向更高回报靠拢


