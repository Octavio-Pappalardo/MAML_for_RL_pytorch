import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from utils import minigrid_preprocess_obs



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

    
    
class Minigrid_obs_encoder(nn.Module):
    def __init__(self, env, obs_encoding_size: int = 32 ) -> None:
        '''Neural net that encodes an image to a vector. Specifically made for encoding the image observation from the minigrid env

        Args:
            env: The minigrid environment the network will be used for. It is used to automatically determine some parameters
            features_dim : The number of features desired for the observations encoding to have.
        '''
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=1)),
            nn.ELU(),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute num of features after cnn by doing one forward pass
        with torch.no_grad():
            sample_obs=env.observation_space.sample()
            sample_obs=minigrid_preprocess_obs(sample_obs)
            n_features_after_flatten = self.cnn(sample_obs.unsqueeze(0).transpose(1, 3).transpose(2, 3)).shape[1]

        self.linear = nn.Sequential(layer_init(nn.Linear(n_features_after_flatten , obs_encoding_size)), nn.ReLU())


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''Takes in a tensor of shape (B1,B2,H,W,C) and otputs encodings of shape (B1,B2,obs_encoding_size)
        Can also handle inputs of shape  (N,H,W,C) and (H,W,C) .
        '''
        #if (B1,B2,H,W,C)
        if len(obs.shape)==5:
            original_shape = obs.shape
            obs=obs.view(-1, original_shape[2], original_shape[3], original_shape[4])
        
            obs=obs.transpose(-3, -1).transpose(-2, -1) #change from (N,H,W,C) to (N,C,H,W)
            encoding=self.linear(self.cnn(obs))
            encoding=encoding.view(original_shape[0], original_shape[1],-1)
            return encoding
        #(H,W,C)
        elif len(obs.shape)==3:
            obs=obs.transpose(-3, -1).transpose(-2, -1).unsqueeze(0) #change from (H,W,C) to (1,C,H,W)
            return self.linear(self.cnn(obs)).squeeze() #returns (obs_encoding_size)
        #(B,H,W,C)
        else:
            obs=obs.transpose(-3, -1).transpose(-2, -1) #change from (N,H,W,C) to (N,C,H,W)
            return self.linear(self.cnn(obs)) #returns (N,obs_encoding_size)




class Maml_agent(nn.Module):
    def __init__(self, env):
        super(Maml_agent, self).__init__()

        self.image_encoder= Minigrid_obs_encoder(env, obs_encoding_size = 512 )
        
        self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)

    def forward(self, x, action=None):

        hidden = self.image_encoder(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
