#!/usr/bin/env python
# coding: utf-8

# ## Demonstration of the API for the Stage 2 DQN Skeleton V3

# The following two cells are only needed when working on CoLab. <br>Executing the next cell will ask for permission to access your gDrive. <br>You need to grant access to be able to use this code.

# In[ ]:


moduleDirectory='/content/gdrive/My Drive/Colab Notebooks'
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


# The next cell is only needed when working on CoLab.

# In[ ]:


import sys
sys.path.insert(0,moduleDirectory)


# First, you obvisouly need to import the skeleton code

# In[1]:


from stage2skeletonv3 import *


# API Demonstration

# To set up the framework, call prepare_torch wihtout parameters, this yields a model object (the actual torch DQN, which you can but don't need to use later).

# In[7]:


model = prepare_torch()


# A state description is a numpy float array of statespace size

# In[10]:


statespace_size=7


# Here we just generate a completely random state as a dummy

# In[13]:


import numpy as np

state=np.random.rand(1,statespace_size)


# In[15]:


state


# To get all Q values for a specific state call get_gvals with this state descriptor as the argument. This returns an array of floats (one Q value per action)

# In[18]:


qvals = get_qvals(state)
print(qvals)


# If you want only the maximum Q value for a state you can use get_maxQ instead. Note that this returns the float but as a torch tensor!

# In[21]:


get_maxQ(state)


# Nevermind that this is a PyTorch tensor, you can compute wiht this tensor (almost) as if it were a normal float

# In[24]:


4*get_maxQ(state)


# You can, of ocurse, always extract the value as a normal Python value if you need to

# In[27]:


get_maxQ(state).item()


# To perform a learning step, you operate on a minibatch of (state, action, target) triplets. These triplets are passes to the function train_one_step in three separate parallel lists, like so:

# In[44]:


state1 = np.random.rand(1,statespace_size)
state2 = np.random.rand(1,statespace_size)


# In[46]:


action1 = np.random.randint(0,4)
action2 = np.random.randint(0,4)

For the demonstration we generate two next states that are reached (here, also just random)
# In[48]:


state1_next = np.random.rand(1,statespace_size)
state2_next = np.random.rand(1,statespace_size)


# Recall that the TD-target is Reward+DiscountFactor*MaxQ(next state)

# In[63]:


reward1=11
reward2=22
discount=0.9


# In[65]:


TD_target1 = reward1+discount*get_maxQ(state1_next)
TD_target2 = reward2+discount*get_maxQ(state2_next)


# We are now ready to generate a mini batch of 2 transitions (which we would normally retrieve from a replay buffer)

# In[54]:


states=[state1,state2] # note that these are the start states
actions=[action1,action2]
targets=[TD_target1,TD_target2] # we don't need the next states as these have already been processed in the TD targets


# Finally, you also need to pass the discount rate, which is just a float. Note that this is just a technicality as it is not longer used by this function. However, the API is still defined in this way (we may remove this in the next version).

# In[57]:


gamma=0.95


# In[59]:


current_loss = train_one_step(states, actions, targets, gamma)


# This returns the current DQN loss, which you may but don't have to use for monitoring and debugging purposes.

# In[44]:


current_loss


# Finally, you need to copy the prediction model to the target model every now and then. Both models are handled automatically behind the scenes, but you decide on when this update happens using the update_target function.

# In[61]:


update_target()


# In[ ]:




