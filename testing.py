import numpy as np
import gymnasium as gym
# import gym
import math
import random
from itertools import count
import time
import yaml
from a2c import A2C
# from maddpg import MADDPG
import torch as T
from torch.utils.tensorboard import SummaryWriter
import argparse
import os




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', dest='model_path', help='model path',
                        default='model_save/t1/', type=str)
    parser.add_argument('-ec', '--ep_count', dest='ep_count', help='episode count',
                        default=900, type=int)


    ARGS = parser.parse_args()
    print(ARGS)
    return ARGS

ARGS = parse_args() 







def main(n_steps,log = False):

    env = gym.make("LunarLander-v2",render_mode = "human")
    agent = A2C(state_size=env.observation_space.shape[0],
        action_size = env.action_space.n,
        actor_lr=0.0001,
        critic_lr=0.0001, 
        gamma=0.99)
    agent.load_model(ARGS.model_path,ARGS.ep_count)
    # print(agent)


    rewards_epoch = []
    step_count = 0
    while True:

        state,_ = env.reset()
        done = False
        done2 = False
        total_reward = 0
        
       
        steps = 0

        while not done :
            dist, value = agent.step(state)
            action = dist.sample()

            
            next_state, reward, done,done2, info = env.step(action.item())

            total_reward+=reward
            
            log_prob = dist.log_prob(action).unsqueeze(0)
            

          
            
            state = next_state
            steps+=1
            step_count += 1
            env.render()
            if steps > n_steps:
                done = True
        print(f"total_reward :{total_reward}")


    


if __name__ == '__main__':

    main(
        n_steps=200,
        log = False)


    # main(agent_count=2, 
    #                 local_ratio=CONFIG['general']['local_ratio'], 
    #                 max_cycles=CONFIG['general']['max_cycles'],
    #                 batch_size=4, 
    #                 tau=CONFIG['general']['tau'], 
    #                 lr=CONFIG['general']['lr'], 
    #                 gamma=CONFIG['general']['gamma'], 
    #                 buffer_size=CONFIG['general']['buffer_size'],
    #                 max_episodes=500
    #             )

    
