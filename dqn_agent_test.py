import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingDQN

import torch
import torch.nn.functional as F
import torch.optim as optim
import heapq
import time

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, mode):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if mode == 'DDQN':
            print('*****training with DDQN*****')
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        elif mode == 'Duel':
            print('*****training with DuelingDQN*****')
            self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) 
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        next_states = torch.from_numpy(np.vstack([next_state])).float().to(device)
        states = torch.from_numpy(np.vstack([state])).float().to(device)
        actions = torch.from_numpy(np.vstack([action])).long().to(device)
        rewards = torch.from_numpy(np.vstack([reward])).float().to(device)
        dones = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)
        action_values = torch.argmax(self.qnetwork_local(next_states).detach(), dim=1)
        Q_targets_next = self.qnetwork_target(next_states).detach().index_select(1, action_values)
        # Compute Q targets for current states 
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        temp_diff = Q_expected - Q_targets
        #print(temp_diff.data.cpu().numpy().shape)
        self.memory.add(state, action, reward, next_state, done, temp_diff.data.cpu().numpy()[0,:])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #Double DQN
        action_values = torch.argmax(self.qnetwork_local(next_states).detach(), dim=1)
        Q_targets_next = self.qnetwork_target(next_states).detach().index_select(1, action_values)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) * weights
        loss = loss.mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.prob_alpha = 0.5
        self.memory = [] #(priority, experience)
        #self.priorities = []  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, temp_diff):
        """Add a new experience to memory."""
        mem_len = len(self.memory)
        temp_diff = np.abs(temp_diff) + 1e-5
        #print('temp ',temp_diff)
        if mem_len >= BUFFER_SIZE:
            #if temp_diff >= self.priorities[0]:
            if temp_diff >= self.memory[0][0]:
                #self.priorities.append(temp_diff)
                #heapq.heappush(self.priorities, temp_diff)
                #heapq.heappop(self.priorities)
                e = self.experience(state, action, reward, next_state, done)
                #self.memory.append(e)
                heapq.heappush(self.memory, (temp_diff, time.time(), e))
                heapq.heappop(self.memory)
        else:
            e = self.experience(state, action, reward, next_state, done)
            # self.priorities.append(temp_diff)
            # self.memory.append(e)
            #heapq.heappush(self.priorities, temp_diff)
            heapq.heappush(self.memory, (temp_diff, time.time(), e))
    
    def sample(self,beta=0.4):
        """Randomly sample a batch of experiences from memory."""
        # if len(self.buffer) == self.capacity:
        #     prios = self.priorities
        # else:
        #     prios = self.priorities[:self.pos]
        #prios = np.array(self.priorities)
        prios = np.array([prio for prio, time, e in self.memory])
        #print('%'*10,prios)
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        #print(prios.shape)
        #print(probs.squeeze().shape)
        #print(probs.squeeze())
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs.squeeze())
        experiences = [self.memory[idx][2] for idx in indices]
        
        mem_len    = len(self.memory)
        weights  = (mem_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        #experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack([weight for weight in weights])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)