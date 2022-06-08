import numpy as np
import numpy.random as random
import ray
from ray import tune
from ray.rllib.agents import ppo
from pathlib import Path

import fym
from dynamics import RllibEnv
from postProcessing import plot_validation


def config():
    fym.config.reset()
    fym.config.update({
        "config": {
            "env": RllibEnv,
            "env_config": {
                "dt": 0.01,
                "max_t": 5.,
                "solver": "rk4"
            },
            "num_gpus": 0,
            "num_workers": 4,
            # "num_envs_per_worker": 50,
            # "lr": 0.0001,
            # "gamma": 0.9,
            "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
            "gamma": tune.grid_search([0.9, 0.99, 0.999])
            # "actor_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "critic_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "actor_lr": 0.001,
            # "critic_lr": 0.0001,
        },
        "stop": {
            "training_iteration": 2,
        },
        "local_dir": "./ray_results",
        "checkpoint_freq": 1,
        "checkpoint_at_end": True,
    })


def train():
    ## load config
    cfg = fym.config.load(as_dict=True)
    ## train
    analysis = tune.run(ppo.PPOTrainer, **cfg)
    ## save checkpoint path and config data
    trial_logdir = analysis.get_best_logdir(
        metric='episode_reward_mean',
        mode='max'
    )
    checkpoint_paths = analysis.get_trial_checkpoints_paths(trial_logdir)
    parent_path = "/".join(trial_logdir.split('/')[0:-1])
    checkpoint_logger = fym.logging.Logger(
        Path(parent_path, 'checkpoint_paths.h5')
    )
    checkpoint_logger.set_info(checkpoint_paths=checkpoint_paths)
    checkpoint_logger.set_info(config=fym.config.load(as_dict=True))
    checkpoint_logger.close()
    return parent_path


def validate(parent_path):
    _, info = fym.logging.load(
        Path(parent_path, 'checkpoint_paths.h5'),
        with_info=True
    )
    checkpoint_paths = info['checkpoint_paths']
    initials = RllibEnv.compute_init()
    env_config = ray.put(fym.config.load("config.env_config", as_dict=True))
    print("Validating...")
    futures = [sim.remote(initial, path[0], env_config, num=i)
               for i, initial in enumerate(initials)
               for path in checkpoint_paths]
    ray.get(futures)


@ray.remote(num_cpus=6)
def sim(initial, checkpoint_path, env_config, num=0):
    env = Env(env_config)
    agent = ppo.PPOTrainer(env=Env, config={"explore": False})

    agent.restore(checkpoint_path)
    parent_path = Path(Path(checkpoint_path).parent, f"test_{num+1}")
    data_path = Path(parent_path, "env_data.h5")
    env.logger = fym.Logger(data_path)

    obs = env.reset(initial)
    while True:
        action = agent.compute_single_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    plot_validation(parent_path, data_path)


def main():
    config()
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ## To train
    parent_path = train()
    ## Only to validate
    # parent_path = 

    validate(parent_path)
    ray.shutdown()


if __name__ == "__main__":
    main()
