import numpy as np
import random
from collections import deque
import os
import sys

# 获取当前文件目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到路径
sys.path.append(parent_dir)
# 导入
import notebooks.env_skeleton as teacher  # <- 只 import 模块
# 第一行改完后，把所有函数用模块前缀调用
prepare_torch    = teacher.prepare_torch
update_target    = teacher.update_target
get_qvals        = teacher.get_qvals
get_maxQ         = teacher.get_maxQ
train_one_step   = teacher.train_one_step

class DQNAgent:
    def __init__(self, state_size=15, batch_size=32):
        """
        初始化DQN智能体
        参数:
            state_size: 状态空间大小
            batch_size: 批量训练大小
        """
        # 设置状态和动作空间
        self.state_size = state_size
        self.action_size = 4  # 四个方向：北、南、西、东
        self.batch_size = batch_size
        
        # 创建经验回放缓冲区
        self.memory = deque(maxlen=10000)
        
        # 设置超参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率起始值
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减系数
        
        # 添加学习步数计数器（用于目标网络更新）
        self.learn_step = 0
        
        # 真正修改模块内的常量
        teacher.statespace_size = state_size   # <<< 关键行放这里
        
        # 调用老师提供的函数准备PyTorch环境和网络
        # 注意：这个函数已经帮我们创建和初始化了网络，无需我们手动操作
        self.model = prepare_torch()
        
    def remember(self, state, action, reward, next_state):
        """将经验存储到回放缓冲区"""
        self.memory.append((state, action, reward, next_state))
    
    def choose_action(self, state, epsilon=None):
        """使用ε-贪心策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # 探索：以epsilon的概率随机选择动作
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        
        # 确保状态是numpy数组
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)
        
        # 利用：选择Q值最大的动作
        q_values = get_qvals(state)
        # 确保q_values是numpy数组
        if not isinstance(q_values, np.ndarray):
            q_values = q_values.detach().numpy()
        return np.argmax(q_values)
    
    def learn(self, s, a, r, s2):
        """
        单步学习（类似于QTableAgent接口）
        实际实现中，这个方法会记住经验并在条件满足时批量学习
        """
        # 将经验添加到回放缓冲区
        self.remember(s, a, r, s2)
        
        # 如果缓冲区中有足够的样本，就进行批量学习
        if len(self.memory) >= self.batch_size:
            self.replay()
            
    def replay(self):
        """从经验回放缓冲区中随机采样并更新网络"""
        # 从记忆中随机抽取一个小批量
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 提取经验
        states = []
        actions = []
        targets = []
        
        for state, action, reward, next_state in minibatch:
            # 将tuple转换为numpy数组
            if isinstance(state, tuple):
                state = np.array(state, dtype=np.float32)
            if isinstance(next_state, tuple):
                next_state = np.array(next_state, dtype=np.float32)
            
            # 计算TD目标
            td_target = reward + self.gamma * get_maxQ(next_state)
            
            # 添加到批处理中
            states.append(state)
            actions.append(action)
            targets.append(td_target.item())  # 将PyTorch张量转换为Python标量
        
        # 使用老师提供的函数批量更新网络
        loss = train_one_step(states, actions, targets, self.gamma)
        
        # 更新学习步数计数器并检查是否需要更新目标网络
        self.learn_step += 1
        if self.learn_step % 1000 == 0:
            update_target()
            
        return loss
    
    def update_target_network(self):
        """更新目标网络（现在通常不直接调用，而是在replay中自动更新）"""
        update_target()
        
    def set_epsilon(self, epsilon):
        """设置新的探索率（由trainer控制）"""
        self.epsilon = max(self.epsilon_min, epsilon)
