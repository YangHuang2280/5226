from env import MultiAgentGridEnv, np
import time
import pygame

def demo():
    """演示可视化器"""
    # 创建环境和可视化器
    env = MultiAgentGridEnv()
    vis = Visualizer(env, cell_size=100, fps=1)
    
    # 重置环境
    vis.reset()
    vis.render()
    time.sleep(1)  # 暂停一秒，查看初始状态
    
    # 运行10步
    running = True
    step_count = 0
    max_steps = 20
    total_collisions = 0
    
    try:
        while running and step_count < max_steps:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 生成随机动作
            actions = {i: np.random.randint(4) for i in range(env.num_agents)}
            
            # 执行一步并展示动画
            _, rewards, _, _, step_collisions = vis.animate(actions)
            total_collisions += step_collisions
            
            # 显示奖励和碰撞
            collision_penalty = -5 * step_collisions if step_collisions > 0 else 0
            print(f"步骤 {step_count+1}, 奖励: {rewards}, 碰撞: {step_collisions}, 碰撞惩罚: {collision_penalty}")
            
            step_count += 1
        
        print(f"总碰撞次数: {total_collisions}")
            
    finally:
        vis.close()
