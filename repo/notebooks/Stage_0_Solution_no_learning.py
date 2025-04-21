#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Define the Q-table agent - currently just a hook without learning
class QTableAgent:

    def choose_action(self, state):
        # hook for the policy
        return np.random.randint(num_actions)


# Define the grid world environment
class GridWorldEnvironment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.agent_position = (0, 0)  # Start at the top-left corner of the grid world
        self.reached_A = False
        # fixed food source location
        # self.food_source_location = (4, 4)
        # dynamic changing food source location
        self.food_source_location = (np.random.randint(n), np.random.randint(m))
        self.nest_location = (n-1,m-1)
        self.rewards = np.zeros((grid_rows, grid_cols))
        self.rewards[self.food_source_location[0],self.food_source_location[1]] = 10
        self.rewards[self.nest_location[0],self.nest_location[1]] = 50
     
    
    def _reset(self):
        self.agent_position = (0, 0)  # Start at the top-left corner of the grid world
        self.reached_A = False
        #random the location of food source per episode; commands off this line for the fix food source
        self.food_source_location = (np.random.randint(self.n), np.random.randint(self.m))
        self.rewards = np.zeros((grid_rows, grid_cols))
        self.rewards[self.food_source_location[0],self.food_source_location[1]] = 10
        self.rewards[self.nest_location[0],self.nest_location[1]] = 50
      
 
    def get_state(self):

        reached_A_state = 1 if self.reached_A else 0
        return self.agent_position[0], self.agent_position[1], self.food_source_location[0], self.food_source_location[1], reached_A_state

    def check_done(self):
        if self.agent_position == self.nest_location and self.reached_A:
            return True
        else:
            return False
    
    def take_action(self, action):
        row, col = self.agent_position

        # Perform the chosen action and observe the next state and reward
        if action == 0:  # Up
            next_position = (max(row - 1, 0), col)
        elif action == 1:  # Down
            next_position = (min(row + 1, grid_rows - 1), col)
        elif action == 2:  # Left
            next_position = (row, max(col - 1, 0))
        elif action == 3:  # Right
            next_position = (row, min(col + 1, grid_cols - 1))
       
        if self.reached_A == False: 
            if next_position == self.food_source_location:
                self.reached_A = True
                reward = self.rewards[next_position]
            else:
                 reward = -1
        elif self.reached_A and next_position == self.nest_location:
            reward = self.rewards[next_position]
        
        else:
            reward = -1
            
        self.agent_position = next_position
        return reward



# In[5]:


from time import process_time

t = process_time()

num_episodes = 5000
max_steps = 100 

# Define the grid world dimensions
grid_rows = 5
grid_cols = 5

# Define the number of actions (up, down, left, right)
num_actions = 4

# Create the Q-table agent and grid world environment
agent = QTableAgent()
environment = GridWorldEnvironment(grid_rows,grid_cols)

reward_total = []

for episode in range(num_episodes+1):   
    environment._reset()
    number_of_steps = 0
    reward_per_episode =  0
    while number_of_steps<= max_steps and (environment.agent_position != environment.nest_location or environment.reached_A != True):  
        # Continue until reaching location B or too many steps   
        state = environment.get_state()
        action = agent.choose_action(state)
        reward = environment.take_action(action)
        reward_per_episode += reward
        done = environment.check_done()
        number_of_steps += 1
    reward_total.append(reward_per_episode)
    if (episode%5000==0):
        print(episode, "episodes")

elapsed_time = process_time() - t


print("Finished in ", elapsed_time, " seconds.")


# In[ ]:




