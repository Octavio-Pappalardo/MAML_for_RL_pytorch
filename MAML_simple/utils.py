import torch
import numpy as np



def minigrid_preprocess_obs(obs , normalizing_factor=10):
    #initial format of an image of minigrid is shape (H,W,3) and uint8 (int with max 256 value but usually much smaller)
    obs=obs['image']
    obs = np.array(obs) # Bug of Pytorch: very slow if not first converted to numpy array
    return torch.tensor(obs, dtype=torch.float) / normalizing_factor


class Logs_and_stats:
    def __init__(self):
        self.base_policy_episode_returns=[]
        self.adapted_policies_episode_returns=[]

        self.list_rewards_means=[]
        self.rewards_mean=0

    def update_statistics(self):
        if len(self.list_rewards_means)>90:
            self.list_rewards_means=self.list_rewards_means[-90:]
        self.rewards_mean=np.array(self.list_rewards_means).mean()
