import torch


class Data_buffer:
    def __init__(self,num_steps ,env):
        #Initialize space for storing data
        self.observations= torch.zeros( (num_steps,) + env.observation_space['image'].shape)
        self.actions = torch.zeros((num_steps,) + env.action_space.shape)
        self.logprob_actions = torch.zeros((num_steps))
        self.rewards = torch.zeros((num_steps))
        self.dones = torch.zeros((num_steps))

        self.advantages = torch.zeros((num_steps))
        self.returns = torch.zeros((num_steps))

        #observations[i] stores the obs in step i
        #action[i] stores the action the agent took in step i
        #logprob_actions[i] stores the logrpob action[i] had
        #rewards[i] stores the reward the agent got for doing action[i] from observation[i]
        #Dones[i] stores whether when taking step[i-1] the env was terminated or tuncated.
        #in other words, it says wether the env was reset befor step i. In which case observation[i] is the first obs from the new episode
        self.num_steps=num_steps

    def store_step_data(self,step_index, obs, act, reward, logp,prev_done):
        self.observations[step_index]=obs
        self.actions[step_index]=act
        self.logprob_actions[step_index]=logp
        self.rewards[step_index]=reward
        self.dones[step_index]=prev_done


    def calculate_returns_and_advantages(self, mean_reward=None , gamma=0.95):
        '''calculate an advantage estimate and a return to go estimate for each state in the batch .
          It estimates it using montecarlo and adds a baseline that is calculated using an estimate of the mean reward the agent receives at each step  '''
        baseline=torch.zeros((self.num_steps))

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_baseline=0
                nextnonterminal=0
                next_return=0
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                next_return = self.returns[t + 1]
                next_baseline=baseline[t+1]

            baseline[t]= mean_reward + gamma *  nextnonterminal * next_baseline
            self.returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
        
        self.advantages = self.returns - baseline