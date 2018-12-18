from unityagents import UnityEnvironment
from ddpg_agent_p3 import Agent as ddpg_agent_p3
import torch
import numpy as np



def test():
    # set hyperparameters (not really important for running the agent)

    # set hyperparameters
    buffer_size = int(1e6)  # replay buffer size
    batch_size = 128        # minibatch size
    gamma = 0.99            # discount factor
    tau = 2e-4              # for soft update of target parameters
    lr_actor = 1e-3         # learning rate of the actor
    lr_critic = 1e-3        # learning rate of the critic
    weight_decay = 0        # L2 weight decay
    eps_start = 6
    eps_end = 0
    eps_decay = 250
    actor_fc1_units = 256   # actor network, size of fully connected layer 1
    actor_fc2_units = 128   # actor network, size of fully connected layer 2
    critic_fc1_units = 256  # critic network, size of fully connected layer 1
    critic_fc2_units = 128  # critic network, size of fully connected layer 2
    seed = 0


    ############ THE ENVIRONMENT ###############
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86', seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get the number of agents
    num_agents = len(env_info.agents)

    # get the size of the action space
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]


    # initialize DDPG agent
    ddpg_agent_1 = ddpg_agent_p3(state_size=state_size,
                                 action_size=action_size,
                                 num_agents=1,
                                 random_seed=seed,
                                 buffer_size=buffer_size,
                                 batch_size=batch_size,
                                 gamma=gamma,
                                 tau=tau,
                                 lr_actor=lr_actor,
                                 lr_critic=lr_critic,
                                 weight_decay=weight_decay,
                                 eps_start=eps_start,
                                 eps_end=eps_end,
                                 eps_decay=eps_decay,
                                 actor_fc1_units=actor_fc1_units,
                                 actor_fc2_units=actor_fc2_units,
                                 critic_fc1_units=critic_fc1_units,
                                 critic_fc2_units=critic_fc2_units
                                 )

    ddpg_agent_2 = ddpg_agent_p3(state_size=state_size,
                                 action_size=action_size,
                                 num_agents=1,
                                 random_seed=seed,
                                 buffer_size=buffer_size,
                                 batch_size=batch_size,
                                 gamma=gamma,
                                 tau=tau,
                                 lr_actor=lr_actor,
                                 lr_critic=lr_critic,
                                 weight_decay=weight_decay,
                                 eps_start=eps_start,
                                 eps_end=eps_end,
                                 eps_decay=eps_decay,
                                 actor_fc1_units=actor_fc1_units,
                                 actor_fc2_units=actor_fc2_units,
                                 critic_fc1_units=critic_fc1_units,
                                 critic_fc2_units=critic_fc2_units
                                 )

    ddpg_agent_1.actor_local.load_state_dict(torch.load('checkpoint_actor_1-final.pth'))
    ddpg_agent_1.critic_local.load_state_dict(torch.load('checkpoint_critic_1-final.pth'))
    ddpg_agent_2.actor_local.load_state_dict(torch.load('checkpoint_actor_2-final.pth'))
    ddpg_agent_2.critic_local.load_state_dict(torch.load('checkpoint_critic_2-final.pth'))

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    states = np.reshape(states, (1, 48))
    scores = np.zeros(num_agents)

    for i in range(600):
        action_agent_1 = ddpg_agent_1.act(states, add_noise=False)
        action_agent_2 = ddpg_agent_2.act(states, add_noise=False)
        actions = np.concatenate((action_agent_1, action_agent_2), axis=0)
        actions = np.reshape(actions, (1, 4))
        env_info = env.step(actions)[brain_name]  # send all actions to the environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        next_states = np.reshape(next_states, (1, 48))
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        ddpg_agent_1.step(0, states, actions, rewards[0], next_states, dones)
        ddpg_agent_2.step(1, states, actions, rewards[1], next_states, dones)
        scores += rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step


if __name__ == '__main__':
    test()
