from env import MultiAgentGridEnv
from agent_qtable import QTableAgent, np

def train(steps=1000, render=False):
    # 创建环境和智能体
    env = MultiAgentGridEnv()
    agents = [QTableAgent() for _ in range(4)]
    
    # 重置环境
    obs = env.reset()
    
    # 统计数据
    total_rewards = 0
    total_collisions = 0
    
    # 初始化epsilon值
    epsilon = 0.1
    
    # 训练循环
    for step in range(steps):
        # epsilon衰减
        epsilon = max(0.01, epsilon*0.995)
        
        # 所有智能体选择动作
        actions = {i: agents[i].choose_action(obs[i], epsilon) for i in range(4)}
        
        # 执行动作
        next_obs, rewards, done, info = env.step(actions)
        
        # 步长惩罚
        for aid in rewards:
            rewards[aid] -= 0.1
        
        # 计算总奖励和碰撞
        step_reward = sum(rewards.values())
        total_rewards += step_reward
        
        # 本步碰撞数
        step_collisions = info["collisions"]
        total_collisions += step_collisions
        
        # 每个智能体学习
        alpha = 0.1  # 初始学习率
        for i in range(4):
            # 可选：α衰减
            alpha = max(0.05, alpha*0.999)
            agents[i].learn(obs[i], actions[i], rewards[i], next_obs[i], alpha)
        
        # 更新状态
        obs = next_obs
        
        # 可选的渲染
        if render and step % 100 == 0:
            env.render()
            print(f"步骤 {step}, epsilon: {epsilon:.3f}, 碰撞: {step_collisions}, 总碰撞: {total_collisions}, 奖励: {step_reward}")
    
    print(f"训练完成！总步数: {steps}, 总奖励: {total_rewards}, 总碰撞: {total_collisions}")
    return agents

if __name__ == "__main__":
    # 训练1000步
    agents = train(1000)
    print("Finished 1000 steps.")
