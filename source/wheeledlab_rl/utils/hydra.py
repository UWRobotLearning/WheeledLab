import functools
from typing import Dict
from collections.abc import Callable

from typing import Any

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf, MISSING
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.utils.dict import update_class_from_dict
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from wheeledlab_rl.configs import *
import wheeledlab_rl.configs as configs

cs = ConfigStore.instance()

def _consolidate_resolved_cfgs(run_cfg: RunConfig):
    assert run_cfg.env is not MISSING, "Environment configuration is missing"
    assert run_cfg.agent is not MISSING, "Agent configuration is missing"

    ####### MODIFY CONFIGS USING EXPOSED OVERRIDES ####### TODO: anyway to resolve these better?
    run_cfg.env.scene.num_envs = run_cfg.env_setup.num_envs
    run_cfg.env.seed = run_cfg.agent.seed
    run_cfg.env.sim.device = run_cfg.train.device

    log = run_cfg.train.log
    if log.test_mode:
        log.no_log = True
        log.no_wandb = True
        log.video = False
        log.no_checkpoints = True


def rl_run_cfg_from_dict(run_cfg:DictConfig, run_config_name: str, cfg: Dict[str, Any], env_cfg_class=None, agent_cfg_class=None) -> RunConfig:
    '''Returns the RunConfig object from the dictionary representation of the configclass object. Used
    to recover @property values from composed configs
    TODO: implement for arbitrary configclasses
    '''

    # Fill default run config with train, env, and agent of loaded config
    update_run_cfg: RunConfig = getattr(configs, run_config_name)() # default run config from module
    update_class_from_dict(update_run_cfg.train, run_cfg.train)
    update_run_cfg.env_setup.from_dict(cfg['env_setup'])
    update_run_cfg.agent_setup.from_dict(cfg['agent_setup'])
    update_run_cfg.train.from_dict(cfg['train'])

    # Construct configclasses for missing types
    if env_cfg_class:
        update_run_cfg.env = env_cfg_class()
        update_run_cfg.env.from_dict(cfg['env'])
    else:
        update_run_cfg.env = cfg['env']

    if agent_cfg_class:
        update_run_cfg.agent = agent_cfg_class()
        update_run_cfg.agent.from_dict(cfg['agent'])
    else:
        update_run_cfg.agent = cfg['agent']

    return update_run_cfg


def register_run_to_hydra(run_config_name: str, node: Any):
    """Load the configurations from the registry and update the Hydra configuration store."""
    # register the task to Hydra
    cs.store(name=run_config_name, node=node)

    run_cfg = cs.repo.get(run_config_name + ".yaml").node
    task_name = run_cfg.env_setup.task_name
    agent_cfg_entry_point = run_cfg.agent_setup.entry_point

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)

    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict):
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    env_cfg_dict = replace_slices_with_strings(env_cfg_dict)
    agent_cfg_dict = replace_slices_with_strings(agent_cfg_dict)

    run_cfg.env = env_cfg_dict
    run_cfg.agent = agent_cfg_dict
    cs.store(name=run_config_name, node=run_cfg)

    return env_cfg, agent_cfg


# def hydra_run_config(run_config_name:str, node:Any, auto_resolve_conflicts=True) -> Callable:
def hydra_run_config(run_config_name:str, auto_resolve_conflicts=True) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # env_cfg, agent_cfg = register_run_to_hydra(run_config_name, node)

            # Load configs from registries using run config name
            run_cfg = cs.repo.get(run_config_name + ".yaml").node
            if run_cfg is None:
                raise ValueError(f"Run config {run_config_name} not found in the Hydra registry.")

            task_name = run_cfg.env_setup.task_name
            env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
            agent_cfg = load_cfg_from_registry(task_name, run_cfg.agent_setup.entry_point)

            # define the new Hydra main function
            @hydra.main(config_name=run_config_name, version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg,
                           run_cfg=run_cfg, run_config_name: str=run_config_name):

                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)

                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)

                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])

                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                replace_strings_with_env_cfg_spaces(env_cfg)

                # call the original function
                # run_cfg = node()
                # run_cfg._from_dict(hydra_env_cfg, env_cfg_class=env_cfg.__class__,
                #                    agent_cfg_class=agent_cfg.__class__)
                run_cfg = rl_run_cfg_from_dict(run_cfg, run_config_name, hydra_env_cfg, env_cfg_class=env_cfg.__class__,
                                          agent_cfg_class=agent_cfg.__class__)

                # Resolve interdependencies between various config params (e.g. env.num_envs = env_setup.num_envs)
                if auto_resolve_conflicts:
                    _consolidate_resolved_cfgs(run_cfg)

                func(run_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator
