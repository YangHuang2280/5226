import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MultiAgentGridEnv

def test_reset_shapes():
    env = MultiAgentGridEnv()
    obs = env.reset(123)
    assert len(obs) == 4  
    assert len(obs[0]) == 15  # (x,y,carry,A,B) + 8 sensor bits

def test_headon():
    env = MultiAgentGridEnv()
    # 设置智能体位置
    env.positions = [(0,0), (0,1), (4,4), (4,3)]
    # 设置智能体方向，0向南，1向北，形成头对头碰撞
    env.directions = [(0,1), (0,-1), (0,0), (0,0)]  # 0 向南, 1 向北 -> 相撞
    # 执行碰撞检测
    env._detect_collisions()
    # 验证是否检测到1次碰撞
    assert env.collisions == 1
