import numpy as np
import math
import random

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from model import Actor, Critic
import matplotlib.pyplot as plt



class A2C:
    def __init__(self, state_size,action_size, actor_lr=1e-3, critic_lr = 1e-3, gamma=0.95) -> None:
        

        self.state_size = state_size
        self.action_size = action_size
        
#         self.state_size = 2
#         self.action_size = 4
        
       
        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size, self.action_size)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr = critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = actor_lr)
        
        
        self.log_p = None
        self.gamma = gamma
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        pass



    
    def step(self, observation):
        observation = T.tensor(observation).to(self.device)
#         print(observation)
        dist = self.actor(observation)
#         action = dist.sample()
        
        value = self.critic(observation)
#         print(dist, value)
        return dist, value
        

    def compute_returns(self,next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
        
    def learn(self, ep, log_probs, values, rewards, masks, next_state):
        next_state = T.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)
        
        rewards = T.FloatTensor(rewards).to(self.device)
        masks = T.FloatTensor(masks).to(self.device)
        
        
        returns = self.compute_returns(next_value, rewards, masks, self.gamma)
        
        log_probs = T.cat(log_probs).to(self.device)
        returns = T.cat(returns).detach().to(self.device)
        values = T.cat(values).to(self.device)
        
        advantage = returns - values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()


    
    def render(self):
        plt.plot(self.total_reward_ep)
        plt.title("Agent: " + str(self.agent_idx)+ " Reward Plot")
        plt.xlabel("episodes")
        plt.ylabel("reward score")
        plt.show()

    def save_model(self,model_path,ep):
        # self.qnetwork_local.save('./savedModels/'+str(ep)+'-'+self.model_file)
        T.save(self.actor.state_dict(),"{}/actor_{}".format(model_path,ep))
        T.save(self.critic.state_dict(),"{}/critic_{}".format(model_path,ep))

    def load_model(self,path,ep):
        
        self.actor.load_state_dict(T.load(f"{path}/actor_{ep}",map_location=self.device))
        self.critic.load_state_dict(T.load(f"{path}/critic_{ep}",map_location=self.device))
        self.actor.eval()