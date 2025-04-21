# 多智能体深度Q学习项目

这个项目实现了一个多智能体系统，使用深度Q网络（DQN）进行协调学习，以完成物品运输任务。

## 项目结构

- `env.py`: 5x5网格世界环境实现
- `agent_dqn.py`: 深度Q网络智能体实现
- `trainer_dqn.py`: 训练脚本
- `notebooks/`: 包含老师提供的骨架代码

## 特点

- 多智能体协调避免碰撞
- 共享深度Q网络
- 经验回放机制
- 目标网络定期更新

## 使用方法

```bash
# 训练智能体
python -m repo.trainer_dqn

# 测试智能体
python -m repo.run_dqn
``` 