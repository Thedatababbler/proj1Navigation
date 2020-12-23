### Report

[image1]:https://github.com/Thedatababbler/proj1Navigation/blob/main/DDQN.JPG
[image2]:https://github.com/Thedatababbler/proj1Navigation/blob/main/deul_avg.JPG
[image3]:https://github.com/Thedatababbler/proj1Navigation/blob/main/priority_exp.png
[image4]:https://github.com/Thedatababbler/proj1Navigation/blob/main/duel.png

### Learning Algorithm
For this project, Except implementing the vanilla DQN network and agent, I have also implmented two major variant of the original DQN network and one improved training technique called priority experience replay. 

#### Double DQN(Double DQN)
The differences between Double DQN and vanilla DQN is quite simple, instead of using the same online network both to select and to evaluate an action, we select the action maximizing the 
action-value with one network and evaluate the action-value with another independent network. The motivation behind is beacuse to avoid overoptimistic evaluation. To read more about the
overoptism of DQN, please refer to this research [paper](https://arxiv.org/pdf/1509.06461.pdf). 

In a nutshell, for the vanilla DQN network, we use a target nework to select and to evaluate the actions, but we need to change it to select actions with our local network while evaluating actions with the target network. The changed code is as following:
```javascript
    action_values = torch.argmax(self.qnetwork_local(next_states).detach(), dim=1)
    Q_targets_next = self.qnetwork_target(next_states).detach().index_select(1, action_values)
```

#### Dueling DQN
The core idea of Dueling DQN is to exploring some states without expanding all the following actions with that state. There are some scenario in a game that we simply know that we will recieve a poor scores no matter which action we take.
Thus, to avoid expanding all actions with every state, we can first evaluate an expected state value V(s). And under this state, we can evaluate the 'advantage' of each action as A(s,a). 
Then the Q(s) = V(s) + A(s,a). 

However, during implementation, a simple additive operation could be 'unidentifiable' to the neural network. Thus we need to twist the original form to:
Q(s) = V(s) + (A(s, a) - mean(A(s,a)))

To adapt the original model to do dueling learning, we need to change the following sections.

First, except the two fully connection layers, we need to add two stream, the one is the value_stream as following:
```javascript
  self.value_stream = nn.Sequential(
          nn.Linear(stream_units, stream_units),
          nn.ReLU(),
          nn.Linear(stream_units, 1)
      )
```

And the other one is the advantage stream represented A(s,a):
```javascript
  self.advantage_stream = nn.Sequential(
          nn.Linear(stream_units, stream_units),
          nn.ReLU(),
          nn.Linear(stream_units, stream_units)
      )
```

During the foward section of our model, we need to add the following code to calculate Q(s) = V(s) + (A(s, a) - mean(A(s,a))):
```javascript
    V = self.value_stream(x)
    A = self.advantage_stream(x)
    Q = V + (A - A.mean())
    return Q
```

### Plot of Rewards
Both twisted network has shown a superiority performance in converging time comparing to the vanilla DQN network bench mark. For the Double DQN(DDQN) network, the agent
managed to solve the task, reaching an average scores more than 13.0 in 100 consecutive episodes, in about 1100 episodes, while the vanilla DQN network trained agent took more than
1800 episodes to solve the task.

![DDQN][image1]

On the other hand, the Dueling DQN, which is another variant of the original DQN, could solve the task in 600 episodes.

![Duel][image2]
![Duel2][image4]

### Ideas for Future Work
Except the two major variant of the vanilla DQN network, I have also tried to implement the priority experience replay technique, however, my expeirement did not deliver the results I expected. Comparing to the vanilla exp replay, my implementation of priority replay does boost the scores in the early agent,
but the agent is unable to solve the task even after 2000 episodes. It only achieved a max average scores about 11.0, which is not enough to be considered solved the task. This could because of my implementation is not 
correct. And I will try to repair this problem in future updates.

![priority][image3]
