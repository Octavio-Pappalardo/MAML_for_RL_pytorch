
import numpy as np
import ray

import gymnasium as gym
import numpy as np
import torch
import torchopt
import metaworld
import time


from maml_config import get_config
from network import Maml_agent
from data_collection import collect_data_from_env
from TRPO_and_adapt_loss import policy_loss_for_adaptation 
from data_buffers import Lifetime_buffer




class Evaluation_buffer:
    def __init__(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]

    def collect_data(self, lifetime_return_per_episode ,lifetime_success_per_episode):
        self.lifetimes_returns_per_episode.append(lifetime_return_per_episode)
        self.lifetimes_success_per_episode.append(lifetime_success_per_episode)

    def combine_data(self):
        self.lifetimes_returns_per_episode=np.array(self.lifetimes_returns_per_episode)
        self.lifetimes_success_per_episode=np.array(self.lifetimes_success_per_episode)
        #each is a numpy array of size (num_lifetimes , num_episodes_per_lifetime)

    def clean_buffer(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]





def eval_inner_loop(maml_model_path, benchmark_name, task , config):

    ####--- Set Up ----#####

    # Construct the appropriate benchmark and a method to create an environment with the task specified in the argument 'task'
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{ml1_env_name}', seed=config.seed)
        def create_env(task):
            env = benchmark.train_classes[f'{task.env_name}']() 
            env.set_task(task)  
            env = gym.wrappers.ClipAction(env)
            if config.seeding==True:
                env.action_space.seed(config.seed)
                env.observation_space.seed(config.seed)
            return env
    else:
        benchmark = metaworld.ML10(seed=config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=config.seed)
        def create_env(task):
            benchmark_envs=benchmark.train_classes.copy()
            benchmark_envs.update(benchmark.test_classes)
            env=benchmark_envs[f'{task.env_name}']()
            env.set_task(task)
            env = gym.wrappers.ClipAction(env)
            if config.seeding==True:
                env.action_space.seed(config.seed)
                env.observation_space.seed(config.seed)
            return env

    env=create_env(task)
    
    maml_agent=Maml_agent(env)
    data=torch.load(maml_model_path)
    maml_agent.load_state_dict(data['maml_model_state_dict'])
    statistics=data['statistics']

    inner_optimizer =torchopt.MetaSGD(maml_agent, lr=config.adaptation_lr)

    adaptation_mean_reward_for_control_variate= statistics['rewards_mean']

    lifetime_buffer=Lifetime_buffer(config.num_lifetime_steps ,config.num_env_steps_per_adaptation_update, env ,device='cpu')


    # ADAPTATION - K gradient steps
    # for each adaptation step : collect data with current policy (starting with the base policy) for doing the inner loop adaptation. Use this data to
    # Compute gradient of the tasks's policy objective function with respect to the current policy parameters.
    # finally use this gradient to update the parameters. Repeat the process config.num_adaptation_steps times .

    information={'current_state':torch.tensor(env.reset()[0],dtype=torch.float32) , 'prev_done' :torch.ones(1),
            'current_episode_step_num':0, 'current_episode_success':False ,'current_episode_return':0,
            'current_lifetime_step':0 , 'hidden_state':None}

    for _ in range(config.num_adaptation_updates_in_inner_loop):
        buffer, information =collect_data_from_env(agent=maml_agent,env=env,num_steps=config.num_env_steps_per_adaptation_update ,
                                    information=information, config=config  , lifetime_buffer=lifetime_buffer ,
                                     mean_reward_for_baseline=adaptation_mean_reward_for_control_variate)

        il_policy_loss = policy_loss_for_adaptation(agent=maml_agent, buffer=buffer)

        inner_optimizer.step(il_policy_loss)

 

    #LOSSESS AFTER ADAPTATION
    # collect data with adapted policy for evaluation
    information={'current_state':torch.tensor(env.reset()[0],dtype=torch.float32) , 'prev_done' :torch.ones(1) ,
        'current_episode_step_num':0, 'current_episode_success':False ,'current_episode_return':0,
        'current_lifetime_step':0 , 'hidden_state':None}
    after_adaptation_buffer, _ =collect_data_from_env(agent=maml_agent ,env=env, num_steps=config.num_env_steps_for_estimating_maml_loss  , 
                                            information=information, config=config , lifetime_buffer=lifetime_buffer , 
                                             mean_reward_for_baseline=adaptation_mean_reward_for_control_variate )


    return lifetime_buffer.episodes_returns , lifetime_buffer.episodes_successes











def collect_evaluation_data(maml_model_path, n , config ,  benchmark_name='ML1' ,env_name= 'door-close-v2',
                             evaluate_test_tasks=True):
    ''' For a given benchmark and environment name it collects all the test(or train) tasks corresponding to that environment ,
    evaluates the performance of the meta agent when being applied to those tasks and returns data of that evaluation.

    Args:
        maml_model_path : path of the weights of the maml_model to be evaluated
        n : amount of times each evaluation task is runned
        config : configurations for the evaluation run.
        benchmark_name : name of the benchmark to evaluate on .
        env_name : name of the environment to evaluate on .
        evaluate_test_tasks : if false the evaluation is done over the tasks with which the meta agent trained instead of with the test tasks

    Returns:
        lifetimes_returns_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing the return of each episode
        lifetimes_success_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing whether each episode succeded
    '''


    # Construct the benchmark and get all the evaluation tasks (50)
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{env_name}', seed=config.seed)
        if evaluate_test_tasks==True:
            eval_tasks= benchmark.test_tasks
        else:
            eval_tasks= benchmark.train_tasks


    elif benchmark_name=='ML10' or benchmark_name=='ML45':
        benchmark = metaworld.ML10(seed=config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=config.seed)
        if evaluate_test_tasks==True:
            eval_tasks=[task for task in benchmark.test_tasks
                        if task.env_name == env_name]
        else:
            eval_tasks=[task for task in benchmark.train_tasks
                        if task.env_name == env_name]
    else:
        print('benchmark name not supported')
    
    eval_tasks=eval_tasks * n
    eval_buffer=Evaluation_buffer() 



    #Collect the evaluation data
    remote_eval_inner_loop=ray.remote(eval_inner_loop)
    config_ref=ray.put(config)
    inputs=[ (maml_model_path, benchmark_name,task , config_ref ) for task in eval_tasks]

    results=ray.get([remote_eval_inner_loop.options(num_cpus=1).remote(*args) for args in inputs])
    for episodes_returns,episodes_successes in results:
        eval_buffer.collect_data(lifetime_return_per_episode=episodes_returns ,lifetime_success_per_episode=episodes_successes )

    eval_buffer.combine_data()

    return eval_buffer.lifetimes_returns_per_episode , eval_buffer.lifetimes_success_per_episode
    











if __name__=='__main__':
    import re
    import os
    import time

    # define parameters of the evaluation run
    evaluate_test_tasks=True   #wether to perform the evaluation over the training or test tasks
    maml_model_path= "../maml_models/MAML_ML1_button-press-v2__5684752145__best_model.pth" 
    benchmark_name='ML1' 
    ml1_env_name= 'button-press-v2' #only relevant when using ML1 benchmark
    config_setting= 'metaworld'
    config=get_config(config_setting)
    n= 10


    if ray.is_initialized:
        ray.shutdown()
    ray.init()


    # Extract the time from the file name (identifies the model) using regular expression . assumes model_path has a format like "path/example123__1638439876__best_model.pth"
    maml_model_id = re.search(r'(?<=__)\d+', os.path.basename(maml_model_path)).group()
    print(maml_model_id)
    if evaluate_test_tasks==True:
         train_or_test='test'
    else:
         train_or_test='train'


    #get a list of the names of all the environments that will be evaluated (each of them contains 50 tasks with parametric variation)
    if benchmark_name=='ML1':
        envs_to_evaluate=[ml1_env_name]
    else:
        benchmark = metaworld.ML10(seed=config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=config.seed)
        if evaluate_test_tasks==True:
            envs_to_evaluate= [name for name,env_cls in benchmark.test_classes.items()]
        else:
            envs_to_evaluate= [name for name,env_cls in benchmark.train_classes.items()]

    for env_name in envs_to_evaluate:
        start_time=time.time()

        exp_name= f'{benchmark_name}_{env_name}'
        run_name = f"{exp_name}_MAML__{maml_model_id}"
        returns_data_path = f"../maml_eval_data/{run_name}__{train_or_test}_eval_returns.npy"
        successes_data_path = f"../maml_eval_data/{run_name}__{train_or_test}_eval_successes.npy"

        lifetimes_returns_per_episode_data ,lifetimes_success_per_episode_data= collect_evaluation_data(maml_model_path=maml_model_path,
                                                                                     n=n , 
                                                                                     config=config ,  benchmark_name=benchmark_name,env_name= env_name,
                                                                                     evaluate_test_tasks=evaluate_test_tasks)
             

        np.save(returns_data_path ,lifetimes_returns_per_episode_data)
        np.save(successes_data_path ,lifetimes_success_per_episode_data )

        print(f'evaluation data of experiment : {run_name} ready')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data) }')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data[:,-4:]) }')
        print(f'time take in minutes ={(time.time()-start_time)/60}')
