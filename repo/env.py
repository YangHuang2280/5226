import numpy as np
from collections import defaultdict

# 1. 设置常量 / Define constants
GRID_SIZE = 5
NUM_AGENTS = 4
ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W
PICKUP_REWARD = 1      # A点拾取奖励
DELIVERY_REWARD = 10   # B点递送奖励
DEST_REWARD = 100  # 到达目的地的奖励
STEP_PENALTY = -0.1  # 每步惩罚
COLLISION_PENALTY = -5  # 碰撞惩罚

# 定义周围8个方向的相对坐标
NEIGHBORS = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy)!=(0,0)]

class MultiAgentGridEnv:
    def __init__(self):
        """
        初始化5x5网格世界环境
        """
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
        
        # 初始化网格 (5x5)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # 2. 实现__init__新增 Central-Clock 索引 / Add clock index
        self.clock_idx = 0      # round-robin pointer
        
    def _opposite_agent_at(self, cell, me):
        """
        检查指定单元格是否有方向与当前智能体相反的其他智能体
        """
        for j, pos in enumerate(self.positions):
            # 跳过自己和不在目标单元格的智能体
            if j == me or pos != cell:
                continue
            
            # 需要确认双方 heading # 0
            if self.heading[j] != 0 and self.heading[me] != 0:
                return self.heading[j] == -self.heading[me]
        
        return False

    def _observe(self, i):
        """
        获取智能体i的观察状态，包括8位传感器信息
        
        参数:
            i: 智能体索引
            
        返回:
            包含完整观察信息的元组
        """
        # 获取基本信息：位置和是否携带物品
        x,y = self.positions[i]
        # 创建基本观察向量：[x, y, carry, A_x, A_y, B_x, B_y]
        obs = [x, y, self.carry[i], *self.A, *self.B]
        
        # 添加8位传感器信息
        for dx,dy in NEIGHBORS:
            # 计算邻居单元格的坐标
            nx,ny = x+dx, y+dy
            # 检查坐标是否在网格范围内，且有方向相反的智能体
            if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE and self._opposite_agent_at((nx,ny), i):
                # 有危险，标记为1
                obs.append(1)
            else:
                # 无危险，标记为0
                obs.append(0)
                
        return tuple(obs)
    
    # 3. 实现reset()
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        # 随机 A/B
        self.A = tuple(rng.integers(0, GRID_SIZE, 2))
        while True:
            self.B = tuple(rng.integers(0, GRID_SIZE, 2))
            if self.B != self.A:
                break
        
        # agent 起点随机选 A 或 B
        self.positions = [self.A if i % 2 == 0 else self.B 
                          for i in range(NUM_AGENTS)]
        self.carry = [int(pos == self.A) for pos in self.positions]
        self.directions = [None]*NUM_AGENTS   # 改为None，而非(0,0)
        self.heading = [0]*NUM_AGENTS        # 新增：逻辑方向标记
        self.collisions = 0
        
        return {i: self._observe(i) for i in range(NUM_AGENTS)}

    def step(self, actions):
        """
        执行智能体动作并更新环境状态
        
        参数:
            actions: 包含各智能体动作的字典或列表
        
        返回:
            observations: 各智能体的新观察状态
            rewards: 各智能体获得的奖励
            done: 是否结束
            info: 额外信息
        """
        # 1. 固定执行顺序 / Round-robin
        # 根据当前时钟索引生成执行顺序，保证每个智能体都有机会先行动
        order = [(self.clock_idx + i) % NUM_AGENTS for i in range(NUM_AGENTS)]
        # 更新时钟索引，为下一次执行做准备
        self.clock_idx = (self.clock_idx + 1) % NUM_AGENTS
        
        # 初始化奖励和信息字典
        rewards = {i: STEP_PENALTY for i in range(self.num_agents)}  # 每步惩罚
        info = {"collisions": 0}
        
        # 2. 更新位置 & 方向 / Move each agent
        for agent_id in order:
            # 获取当前智能体的动作
            action_idx = actions[agent_id]
            # 根据动作获取方向变化 (dx, dy)
            dx, dy = ACTIONS[action_idx]
            # 记录智能体的移动方向
            self.directions[agent_id] = (dx, dy)
            
            # 计算新位置，使用np.clip确保位置在网格范围内
            old_x, old_y = self.positions[agent_id]
            new_x = np.clip(old_x + dx, 0, GRID_SIZE-1)
            new_y = np.clip(old_y + dy, 0, GRID_SIZE-1)
            # 更新智能体位置
            self.positions[agent_id] = (new_x, new_y)
            
            # 检查是否到达A或B点，并处理物品拾取和放下
            new_pos = (new_x, new_y)
            if new_pos == self.A and not self.carry[agent_id]:
                # 在A点拾取物品
                self.carry[agent_id] = True
                rewards[agent_id] += PICKUP_REWARD
            elif new_pos == self.B and self.carry[agent_id]:
                # 在B点放下物品
                self.carry[agent_id] = False
                rewards[agent_id] += DELIVERY_REWARD
        
        # 更新逻辑方向
        self._update_heading()
        
        # 检测碰撞并应用惩罚
        step_collisions = self._detect_collisions(rewards)
        info["collisions"] = step_collisions
        
        # 准备返回的观察状态
        observations = {i: self._observe(i) for i in range(self.num_agents)}
        done = False  # 无限任务，永不结束
        
        return observations, rewards, done, info

    def _detect_collisions(self, rewards):
        """
        检测头对头碰撞并返回本步碰撞数
        
        参数:
            rewards: 奖励字典，用于扣除碰撞惩罚
        返回:
            本步发生的碰撞数
        """
        old_collisions = self.collisions
        cell_to_agents = defaultdict(list)
        
        for idx, pos in enumerate(self.positions):
            cell_to_agents[pos].append(idx)
        
        for pos, agents in cell_to_agents.items():
            # 如果单元格只有一个智能体，跳过
            if len(agents) < 2:
                continue
            
            # 检查是否有相反方向的智能体
            headings = {self.heading[a] for a in agents}
            if 1 in headings and -1 in headings:  # 同时有A→B和B→A的智能体
                if pos not in (self.A, self.B):   # 非A/B点的碰撞
                    self.collisions += 1
                    # 给碰撞智能体扣分
                    for a in agents:
                        rewards[a] += COLLISION_PENALTY
        
        # 返回本步新增的碰撞数
        return self.collisions - old_collisions

    def _update_heading(self):
        """根据是否携带物品更新 A→B / B→A 标记"""
        self.heading = []
        for carry, pos in zip(self.carry, self.positions):
            if carry and pos != self.B:   # 正在运送，方向 A→B
                self.heading.append(1)
            elif (not carry) and pos != self.A:  # 空手返回，方向 B→A
                self.heading.append(-1)
            else:
                self.heading.append(0)

    def render(self):
        """
        渲染环境的文本表示
        在终端中显示网格环境状态
        """
        # 创建空白网格
        grid = [["  " for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        # 标记A和B位置
        ax, ay = self.A
        bx, by = self.B
        grid[ay][ax] = " A"
        grid[by][bx] = " B"
        
        # 标记智能体位置
        for i, (x,y) in enumerate(self.positions):
            # 用数字标记智能体，携带物品的智能体会加*号
            marker = f"{i}"
            if self.carry[i]:
                marker += "*"
            grid[y][x] = marker.rjust(2)
        
        # 打印网格
        print("\n".join("".join(row) for row in grid))
