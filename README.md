# proj1Navigation

[image1]:https://github.com/Thedatababbler/proj1Navigation/blob/main/DDQN.JPG

[image3]:https://github.com/Thedatababbler/proj1Navigation/blob/main/priority_exp.png

### Project Details
This is the implementation of a trained agent to navigate in a Unity ML environment 

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive so this task could be considered as solved.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Clone this project to your local directory and place the file in the and same directory, and unzip (or decompress) the file.

3. Please refer to this page (https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up the dependencies.

### Instruction
For this project, Except implementing the vanilla DQN network and agent, I have also implmented two major variant of the original DQN network and one improved training technique called
priority experience replay. 

Both twisted network has shown a superiority performance in converging time comparing to the vanilla DQN network bench mark. For the Double DQN(DDQN) network, the agent
managed to solve the task, reaching an average scores more than 13.0 in 100 consecutive episodes, in about 1100 episodes, while the vanilla DQN network trained agent took more than
1800 episodes to solve the task. To train the agent with DDQN network, please first clone this repository, then run: 
```javascript
python Navigation.py DDQN replay
```
While the first argument indicate which network you prefered to use, the second argument is related to the exp replay technique which I will mention in later section.

![DDQN][image1]

On the other hand, the Dueling DQN, which is another variant of the original DQN, could solve the task in 600 episodes. To train the agent with DDQN network, please first clone this repository, then run: 
```javascript
python Navigation.py Duel replay
```

Experience Replay is a vital training technique for a RL agent, however, a experience replay buffer implemented with FIFO queue structure has its limit. Thus, an improved version of experience replay
is to add an priority for each state. While sampling a batch for training, the priority will dedicate the probability of sampling. I adapt the code in dqn_agent.py to include this improved
version of exp replay. However, my expeirement did deliver the results I expected. Comparing to the vanilla exp replay, my implementation of priority replay does boost the scores in the early agent,
but the agent is unable to solve the task even after 2000 episodes. It only achieved a max average scores about 11.0, which is not enough to be considered solved the task. This could because of my implementation is not 
correct. And I will try to repair this problem in future updates. 

![priority][image3]

To train the agent with DDQN network, please first clone this repository, then run: 
```javascript
python Navigation.py Duel priority
```
you could train both networks with either priority exp replay or the origin exp replay.

__Note: For grading purpose, please train the agent with Dueling DQN without priority exp replay. Run as following__
```javascript
python Navigation.py Duel replay
```

### Training Aftermath
When you finished training, the code will directly saved a **'checkpoint.path'** file, which contains all the weights of the network you just trained, to your current path. You could
use this file to run a simple 10 episodes validation runs to check the trained agent's performance

--TODO





