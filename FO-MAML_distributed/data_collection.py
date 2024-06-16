import torch
from data_buffers import Maml_data_buffer


def collect_data_from_env(agent,env,num_steps , information, config,
                          lifetime_buffer , env_name=None ,for_maml_loss=False , mean_reward_for_baseline=None
                          ,device='cpu'):

    episodes_returns=[]
    max_episode_steps=config.max_episode_steps
    episode_step_num=information['current_episode_step_num']
    episode_return= information['current_episode_return']
    episodes_successes=[] #keeps track of wether the goal was achieved in each episode
    succeded_in_episode=information['current_episode_success'] #keeps track of wether the agent has achieved succes in current episode
    current_lifetime_step=information['current_lifetime_step']
    hidden_state=information['hidden_state']

    #instantiate a buffer to save the data in
    actionA_buffer= Maml_data_buffer(num_steps=num_steps ,env=env ,device=device,env_name=env_name)
    
    #get an initial state from the environment 
    next_obs=information['current_state']
    done = information['prev_done']

    for step in range(0, num_steps):

        obs, prev_done = next_obs, done

        with torch.no_grad():
            action, logprob, _ = agent.get_action(obs.unsqueeze(0))  

        #execute the action and get environment response.
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())  
        done= torch.max(torch.tensor([terminated,truncated],dtype=torch.float32))

        #preprocess and store data 
        reward= torch.as_tensor(reward,dtype=torch.float32).to(device)
        lifetime_buffer.store_step_data(global_step=current_lifetime_step , obs=obs.to(device), act=action.to(device),
                            reward=reward.to(device), logp=logprob.to(device),prev_done=prev_done.to(device))

        actionA_buffer.store_inner_loop_update_data(step_index=step, obs=obs, act=action, reward=reward
                                                    , logp=logprob,prev_done=prev_done)
 

        #prepare for next step
        next_obs, done =torch.as_tensor(next_obs,dtype=torch.float32).to(device) ,torch.as_tensor(done,dtype=torch.float32).to(device)
        current_lifetime_step+=1
        episode_step_num+=1
        episode_return+= reward
        if info['success'] == 1.0: 
            succeded_in_episode=True 



        #deal with the case where the episode ends 
        if episode_step_num==max_episode_steps :
            episodes_returns.append(episode_return)
            if succeded_in_episode==True:
                episodes_successes.append(1.0) 
            else:
                episodes_successes.append(0.0)
            done=torch.ones(1).to(device) 
            next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(device) 
            episode_step_num=0
            episode_return=0
            succeded_in_episode=False

                
    if not for_maml_loss:
        with torch.no_grad():
            #calculate the advantages and to go returns for each state visited in the data. 
            actionA_buffer.calculate_returns_and_advantages(mean_reward=mean_reward_for_baseline,gamma=config.adaptation_gamma)

    lifetime_buffer.episodes_returns=lifetime_buffer.episodes_returns+episodes_returns
    lifetime_buffer.episodes_successes =lifetime_buffer.episodes_successes+ episodes_successes 

    information={'current_state':next_obs , 'prev_done' :done  ,
                'current_episode_step_num':episode_step_num , 'current_episode_success':succeded_in_episode ,'current_episode_return':episode_return,
                'current_lifetime_step':current_lifetime_step ,'hidden_state':hidden_state}
    
    return actionA_buffer , information




