import pygame
import numpy as np
import time
from env import MultiAgentGridEnv
import os

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (200, 200, 200)

class Visualizer:
    def __init__(self, env, cell_size=80, fps=2):
        """
        初始化可视化器
        
        参数:
            env: 网格环境
            cell_size: 单元格大小(像素)
            fps: 每秒帧数
        """
        self.env = env
        self.cell_size = cell_size
        self.fps = fps
        self.grid_size = env.grid_size
        
        # 初始化pygame
        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
        pygame.display.set_caption("多智能体网格世界")
        
        # 创建窗口
        self.screen_width = self.grid_size * self.cell_size
        self.screen_height = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # 用于平滑动画的变量
        self.old_positions = None
        self.animation_steps = 5
        self.current_step = 0
        
        # 字体
        self.font = pygame.font.SysFont('Arial', 18)
        
    def draw_grid(self):
        """绘制网格"""
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.screen_width, y))
    
    def draw_locations(self):
        """绘制A和B位置"""
        ax, ay = self.env.A
        bx, by = self.env.B
        
        # 绘制A位置
        pygame.draw.rect(self.screen, GREEN, 
                        (ax * self.cell_size, ay * self.cell_size, 
                         self.cell_size, self.cell_size))
        
        a_text = self.font.render('A', True, BLACK)
        a_rect = a_text.get_rect(center=(ax * self.cell_size + self.cell_size//2,
                                         ay * self.cell_size + self.cell_size//2))
        self.screen.blit(a_text, a_rect)
        
        # 绘制B位置
        pygame.draw.rect(self.screen, RED, 
                        (bx * self.cell_size, by * self.cell_size, 
                         self.cell_size, self.cell_size))
        
        b_text = self.font.render('B', True, BLACK)
        b_rect = b_text.get_rect(center=(bx * self.cell_size + self.cell_size//2,
                                         by * self.cell_size + self.cell_size//2))
        self.screen.blit(b_text, b_rect)
    
    def draw_agents(self):
        """绘制智能体"""
        agent_colors = [BLUE, YELLOW, PURPLE, (0, 128, 128)]  # 不同智能体不同颜色
        
        for i, (x, y) in enumerate(self.env.positions):
            color = agent_colors[i % len(agent_colors)]
            
            # 绘制智能体
            agent_radius = self.cell_size // 3
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), agent_radius)
            
            # 绘制智能体编号
            agent_text = self.font.render(str(i), True, WHITE)
            agent_rect = agent_text.get_rect(center=(center_x, center_y))
            self.screen.blit(agent_text, agent_rect)
            
            # 如果智能体携带物品，绘制一个星号
            if self.env.carry[i]:
                carry_text = self.font.render('*', True, WHITE)
                carry_rect = carry_text.get_rect(center=(center_x, center_y - agent_radius - 5))
                self.screen.blit(carry_text, carry_rect)
    
    def draw_info(self):
        """绘制信息"""
        # 显示碰撞次数
        collisions_text = self.font.render(f'碰撞: {self.env.collisions}', True, BLACK)
        self.screen.blit(collisions_text, (10, 10))
    
    def animate(self, actions):
        """
        动画展示一步执行过程
        
        参数:
            actions: 智能体的动作
        """
        if self.old_positions is None:
            self.old_positions = self.env.positions.copy()
            
        # 保存旧位置
        old_positions = self.old_positions.copy()
        old_collisions = self.env.collisions  # 记录执行前的碰撞数
        
        # 执行环境步骤
        obs, rewards, done, info = self.env.step(actions)
        
        # 计算这一步发生的碰撞数
        new_collisions = self.env.collisions
        step_collisions = new_collisions - old_collisions
        
        # 设置当前为新位置
        new_positions = self.env.positions.copy()
        
        # 动画展示移动过程
        for step in range(self.animation_steps):
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_locations()
            
            # 计算插值位置
            t = step / self.animation_steps
            interp_positions = []
            
            for i, ((old_x, old_y), (new_x, new_y)) in enumerate(zip(old_positions, new_positions)):
                interp_x = old_x + (new_x - old_x) * t
                interp_y = old_y + (new_y - old_y) * t
                interp_positions.append((interp_x, interp_y))
            
            # 临时替换位置进行绘制
            temp_positions = self.env.positions
            self.env.positions = interp_positions
            self.draw_agents()
            self.env.positions = temp_positions
            
            # 绘制碰撞信息
            self.draw_info()
            
            # 如果这一步有碰撞，在屏幕上显示警告
            if step_collisions > 0:
                collision_text = self.font.render(f'本步发生 {step_collisions} 次碰撞!', True, RED)
                self.screen.blit(collision_text, (10, 40))
            
            pygame.display.flip()
            self.clock.tick(self.fps * 3)  # 动画更快一些
        
        self.old_positions = new_positions
        return obs, rewards, done, info, step_collisions  # 返回碰撞信息
    
    def reset(self):
        """重置环境和可视化器"""
        obs = self.env.reset()
        self.old_positions = None
        return obs
    
    def render(self):
        """渲染当前状态"""
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_locations()
        self.draw_agents()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def close(self):
        """关闭可视化器"""
        pygame.quit()


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

if __name__ == "__main__":
    demo()
