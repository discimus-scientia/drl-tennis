![image](tennis.png)

# Learning to Play Tennis with Deep Reinforcement Learning

## Environment Details
In this environment, two agents control rackets to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, 
it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.


### State Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
Each agent receives its own, local observation.

### Action Space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solution Criterium
The task is episodic. In order to solve the environment, the agents 
must get an average score of +0.5 over 100 consecutive episodes, after taking the maximum of both agents.
Specifically,

- After each episode, we add up the rewards that each agent received (without discounting),
 to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of
 these 2 scores.
 - This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of 
those scores is at least +0.5. 

## How To Run
### Dependencies

1. Download the environment from one of the links below. You need only select
   the environment that matches your operating system:
   
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
2. Place the file in the DRLND GitHub repository, in the
   `p3_collab-compet/` folder, and unzip (or decompress) the file.


### How To Run The Code
Run `python train.py` to start training or `python test.py` to observe how the trained agent performs with 
the weights included.

