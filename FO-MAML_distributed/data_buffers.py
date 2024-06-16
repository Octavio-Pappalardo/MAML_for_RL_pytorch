import torch

class Maml_data_buffer:
    def __init__(self,num_steps ,env,device,env_name='none'):

        self.observations= torch.zeros( (num_steps , env.observation_space.shape[0])).to(device)
        self.actions = torch.zeros((num_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions = torch.zeros((num_steps)).to(device)
        self.rewards = torch.zeros((num_steps)).to(device)
        self.dones = torch.zeros((num_steps)).to(device)
        self.advantages = torch.zeros((num_steps)).to(device)
        self.returns = torch.zeros((num_steps)).to(device)

        self.batch_size = int( num_steps)
        self.num_steps=num_steps
        self.env_name=env_name


    def store_inner_loop_update_data(self,step_index, obs, act, reward, logp,prev_done):
        self.observations[step_index]=obs
        self.actions[step_index]=act
        self.logprob_actions[step_index]=logp
        self.rewards[step_index]=reward
        self.dones[step_index]=prev_done


    def preprocess_data(self, data_stats , objective_mean):
        ''' Normalizes rewards using environment dependant statistics. It multiplies the rewards by a factor that makes the mean equal to objective_mean
        Args:
            data_stats: An object that keeps track of the mean reward given by each environment type
            objective_mean : 
        '''
        if f'{self.env_name}' in data_stats.rewards_means.keys():
            self.rewards= (self.rewards/ (data_stats.rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean


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





class Lifetime_buffer:
    def __init__(self,num_lifetime_steps ,il_batch_size, env, device ,env_name='none'):
        '''class for storing all the data the agent collects throughout an inner loop .'''

        self.observations=torch.zeros( (num_lifetime_steps , env.observation_space.shape[0])).to(device)
        self.actions= torch.zeros((num_lifetime_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions= torch.zeros((num_lifetime_steps)).to(device)
        self.dones= torch.zeros((num_lifetime_steps)).to(device)

        self.rewards= torch.zeros((num_lifetime_steps)).to(device)

        self.device=device
        self.il_batch_size=il_batch_size #batch_size used in the inner loop
        self.num_lifetime_steps=num_lifetime_steps
        self.episodes_returns=[] #list that contains the returns of each episode in the lifetime
        self.episodes_successes=[] #list that contains wether each episode in the lifetime succeded in completing the task 
  
        self.env_name=env_name

    def store_step_data(self,global_step, obs, act,reward, logp,prev_done):
        self.observations[global_step]=obs.to(self.device)
        self.actions[global_step]=act.to(self.device)
        self.logprob_actions[global_step]=logp.to(self.device)
        self.dones[global_step]=prev_done.to(self.device)
        self.rewards[global_step]=reward.to(self.device)
