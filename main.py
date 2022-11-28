import numpy as np
import gymnasium as gym
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
    parser.add_argument('-t', '--trial', dest='trial', help='trial',
                        default="t2", type=str)


    ARGS = parser.parse_args()
    print(ARGS)
    return ARGS

ARGS = parse_args() 







def main(total_episode,n_steps,log = False):
    if log :
        if not os.path.isdir(f"model_save/{ARGS.trial}"):
            os.makedirs(f"model_save/{ARGS.trial}")

        writer = SummaryWriter(f"tensorboard/{ARGS.trial}")

    env = gym.make("LunarLander-v2")
    agent = A2C(state_size=env.observation_space.shape[0],
        action_size = env.action_space.n,
        actor_lr=0.0001,
        critic_lr=0.0001, 
        gamma=0.99)

    # print(agent)

    episodes = total_episode

    rewards_epoch = []
    step_count = 0
    for ep in range(episodes):

        state,_ = env.reset()
        done = False
        done2 = False
        total_reward = 0
        
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        steps = 0

        while not done :
            dist, value = agent.step(state)
            action = dist.sample()

            
            next_state, reward, done,done2, info = env.step(action.item())
            
            total_reward+=reward
            
            log_prob = dist.log_prob(action).unsqueeze(0)
            

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(T.tensor([reward], dtype = T.float))
            masks.append(T.tensor([1-done], dtype = T.float))
            
            state = next_state
            steps+=1
            step_count += 1
            if step_count % n_steps == 0:
                agent.learn(ep, log_probs, values, rewards, masks, next_state)
                log_probs = []
                values = []
                rewards = []
                masks = []
                

        
        # writer.add_scalar(f"loss/actor",actor_loss.item() , ep)
        # writer.add_scalar(f"loss/critic",critic_loss.item() , ep)
        writer.add_scalar(f"reward/total_reward",total_reward , ep)
        
        rewards_epoch.append(total_reward)

        

        if ep % 100 == 0 and ep>0:
            agent.save_model(f"model_save/{ARGS.trial}",ep)
            print('Episode {}'.format(ep))
            print('Avg Reward {}'.format(np.mean(rewards_epoch[-100])))

            print('***************************************************')
    


if __name__ == '__main__':

    main(total_episode=5000,
        n_steps=50,
        log = True)


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

    
