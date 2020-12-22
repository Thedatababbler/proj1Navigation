import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
import argparse
import sys


# load the weights from file
def main():
    scores = 0
    for i in range(10):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        
        while True:
            action = agent.act(state)
            action = action.astype(int)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   
            # get the next state
            reward = env_info.rewards[0]   # get the reward
            done = env_info.local_done[0]
            scores += reward
            state = next_state
            if done:
                break 
        print ('\rEpisode {}\tAverage Score: {:.2f}'.format(i+1, scores/(i+1), end=''))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="you can choose to use DDQN by input DDQN or use DuelingDQN by input Duel here")
    args = parser.parse_args()
    if args.network not in ('DDQN', 'Duel'):
        print('No such network, please make sure you choose either DDQN or Duel as an input for the network argument')
        sys.exit()
    agent = Agent(state_size=37, action_size=4, seed=0, mode=args.network)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    main()