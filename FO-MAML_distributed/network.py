
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Maml_agent(nn.Module):
    def __init__(self, env):
        super(Maml_agent, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(env.action_space.shape)), std=0.01),
        )

        self.actor_logstd=nn.Sequential(
            layer_init(nn.Linear(np.prod(env.observation_space.shape), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=0.01) )

    def get_action(self, x, action=None ,return_distribution=False):
        action_mean = self.actor_mean(x)

        action_logstd= self.actor_logstd(x)
        action_std = torch.exp(action_logstd )
        distribution = Normal(action_mean, action_std)
        if action is None:
            action= distribution.sample() 
            logprob= distribution.log_prob(action).sum(1)
        else:
            logprob= distribution.log_prob(action).sum(1)

        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1) 
        elif return_distribution==True:
            return action, logprob, distribution.entropy().sum(1),  distribution
        

    def get_deterministic_action(self,x):
        action_mean = self.actor_mean(x)
        return action_mean
    



