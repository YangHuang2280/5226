import numpy as np
from collections import defaultdict

class QTableAgent:
    def __init__(self):
        self.Q = {}
        
    def choose_action(self, state, epsilon=0.1):
        # 状态转换为不可变类型，作为字典键
        state = tuple(state)
        
        # epsilon-greedy策略
        if np.random.rand() < epsilon or state not in self.Q:
            return np.random.randint(4)
        
        # 选择Q值最大的动作
        return int(max(self.Q[state], key=self.Q[state].get))
    
    def learn(self, s, a, r, s2, alpha=0.1, gamma=0.95):
        # 状态转换为不可变类型
        s = tuple(s)
        s2 = tuple(s2)
        
        # 确保状态-动作对在Q表中有初始值
        self.Q.setdefault(s, {k:0 for k in range(4)})
        self.Q.setdefault(s2, {k:0 for k in range(4)})
        
        # Q-learning更新公式: Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)]
        td = r + gamma*max(self.Q[s2].values()) - self.Q[s][a]
        self.Q[s][a] += alpha*td
