
class Metaworld:
    def __init__(self):
        self.benchmark_name= 'ML1' 
        self.env_name= 'reach-v2' 
        self.max_episode_steps=500  

        self.num_epsiodes_of_validation = 4
        self.num_lifetimes_for_validation = 120

        self.seeding=False
        self.seed=1
        self.device='cpu'

        self.num_outer_loop_updates=4000
        self.num_inner_loops_per_update=30  #num of tasks considered for each outer loop update

        self.num_adaptation_updates_in_inner_loop=2 
        self.num_env_steps_per_adaptation_update=2000 
        self.num_env_steps_for_estimating_maml_loss=4000
        self.num_lifetime_steps=self.num_adaptation_updates_in_inner_loop * self.num_env_steps_per_adaptation_update + self.num_env_steps_for_estimating_maml_loss

        self.maml_agent_lr=5e-4 
        self.maml_agent_epsilon=1e-5

        self.adaptation_lr= 7e-2 
        self.adaptation_gamma=0.995 

        self.rewards_target_mean_for_maml_agent=  0.1  #normalization
 

        self.maml_agent_gamma=0.995 

        self.maml_TRPO={
            "cg_damping": 1e-2 ,
            "max_kl" :  0.001,
            "cg_iters" : 10,
            "line_search_max_steps": 10,
            "line_search_backtrack_ratio": 0.6
        } 



def get_config(config_settings):
    if config_settings=='metaworld':
        return Metaworld()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
