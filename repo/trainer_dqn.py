from env import MultiAgentGridEnv
from agent_dqn import DQNAgent
import numpy as np
import time

def train_dqn(steps=1000000, render_interval=1000):
    """
    使用DQN训练多智能体系统
    
    参数:
        steps: 训练步数
        render_interval: 渲染频率
    """
    # 创建环境
    env = MultiAgentGridEnv()
    
    # 创建智能体（共享网络）
    state_size = 15  # 7个基本状态 + 8个传感器输入
    shared_agent = DQNAgent(state_size=state_size)
    agents = [shared_agent] * 4  # 所有智能体共享同一个网络
    
    # 探索率参数
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    # 重置环境
    obs = env.reset()
    
    # 统计数据
    total_rewards = 0
    total_collisions = 0
    steps_taken = 0
    start_time = time.time()
    
    # 训练循环
    for step in range(steps):
        # 所有智能体选择动作（使用当前探索率）
        actions = {i: agents[i].choose_action(obs[i], epsilon) for i in range(4)}
        
        # 执行动作
        next_obs, rewards, done, info = env.step(actions)
        
        # 计算总奖励和碰撞
        step_reward = sum(rewards.values())
        total_rewards += step_reward
        
        # 本步碰撞数
        step_collisions = info["collisions"]
        total_collisions += step_collisions
        steps_taken += 4  # 4个智能体各执行一步
        
        # 每个智能体学习（但不在学习中衰减epsilon）
        for i in range(4):
            agents[i].learn(obs[i], actions[i], rewards[i], next_obs[i])
        
        # 衰减探索率（每个回合只衰减一次）
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 更新状态
        obs = next_obs
        
        # 显示训练进度
        if step % render_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"步骤 {step}/{steps} ({step/steps*100:.1f}%), epsilon: {epsilon:.3f}, "
                  f"碰撞: {total_collisions}, 奖励: {total_rewards}, 时间: {elapsed_time:.1f}秒")
            
            # 检查训练预算限制
            if total_collisions >= 4000 or elapsed_time > 600 or steps_taken >= 1500000:  # 训练预算限制
                print("达到训练预算上限！")
                break
                
        # 可选的渲染
        if step % (render_interval * 10) == 0:
            env.render()
    
    # 训练结束统计
    elapsed_time = time.time() - start_time
    print(f"训练完成！总步数: {steps_taken}, 总奖励: {total_rewards}, "
          f"总碰撞: {total_collisions}, 总时间: {elapsed_time:.1f}秒")
    
    return shared_agent

if __name__ == "__main__":
    # 训练智能体
    trained_agent = train_dqn(steps=1500000)  # 最多150万步
    print("训练完成！")
