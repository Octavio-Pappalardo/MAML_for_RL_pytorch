class Config_custom:
    def __init__(self):
        self.max_episode_length=140
        self.env_size=10

        self.num_meta_updates=450
        self.num_tasks_per_meta_update=10
        self.num_adaptation_steps=1

        self.meta_lr=3e-4

        self.il_lr=0.1  #0.0001 when using adam  , 0.1 for SGD ,
        self.env_steps_to_estimate_loss=3000

        self.gamma=0.99


def get_config(config_settings):
    if config_settings=='custom':
        return Config_custom()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
