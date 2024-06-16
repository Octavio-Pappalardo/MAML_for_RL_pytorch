import torch
from data_buffers import Data_buffer
import numpy as np
from torch.func import functional_call
from utils import minigrid_preprocess_obs


def collect_data_from_env(agent, env,num_steps,logger ,config , params=None):
    episodes_lengths=[]
    episodes_returns=[]

    #instantiate a buffer to save the data in
    buffer= Data_buffer(num_steps=num_steps ,env=env )
    
    #get an initial state from the environment 
    next_obs=env.reset()[0] 
    done = torch.zeros(1)

    for step in range(0, num_steps):

        #prepare for new step: next_obs becomes the new step's observation 
        obs, prev_done = next_obs, done

        obs=minigrid_preprocess_obs(obs)

        # get actionA action predictions and state value estimates
        with torch.no_grad():
            if params:
                action, logprob, _  = functional_call(agent,params,obs.unsqueeze(0))
            else:
                action, logprob, _  = agent(obs.unsqueeze(0))  

        #execute the action and get environment response.
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())  
        done= torch.max(torch.tensor([terminated,truncated],dtype=torch.float32))

        #preprocess and store data 
        reward= torch.as_tensor(reward,dtype=torch.float32)

        buffer.store_step_data(step_index=step, obs=obs, act=action.long(), reward=reward
                                                        , logp=logprob,prev_done=prev_done)


        #prepare for next step
        done = torch.as_tensor(done,dtype=torch.float32)


        #deal with the case where the episode ends
        if done:
            #reset environment
            next_obs = env.reset()[0] 
            #save metrics
            assert 'episode' in info , 'problem with recordeEpisodeStatistics wrapper'
            episodes_returns.append(info['episode']['r'][0])
            episodes_lengths.append(info['episode']['l'][0])

                 

    #calculate the advantages and to go returns for each state visited in the data. 
    with torch.no_grad():
        buffer.calculate_returns_and_advantages(mean_reward=logger.rewards_mean,gamma=config.gamma)

    #calculate some metrics for future logging and for keeping track of statistics
    logger.list_rewards_means.append(torch.mean(buffer.rewards))
    mean_episode_return= np.array(episodes_returns).mean()

    return buffer , mean_episode_return


