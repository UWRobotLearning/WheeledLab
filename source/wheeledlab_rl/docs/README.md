# WheeledLab RL

A single RL run contains an overwhelming number of parameters. It is highly advised to read the configuration hierarchy through an IDE that can navigate through class definitions.

## Configs

Wheeled Lab aims to expose all training behavior through Configuration Files (configs). Configs are **human-readable** and importantly make the process of model selection and deployment more scientific. Configs can be thought of as Python `dataclass` types (techncially they are Isaac Lab `configclass` types).

An overview of the config structure can be found below.

```
RunConfig
│
├── TrainConfig
│   ├── LogConfig
│   │   ├── Log Format [W&B|Tensorboard|Offline]
│   │   ├── Model Checkpoints
│   │   └── [...]
│   ├── Algorithm [PPO|SAC]
│   └── Alg. Library [RSL|SB3]
│
├── EnvConfig
│   ├── Actions [Ackermann|4WD|RWD]
│   ├── Rewards
│   ├── Observations
│   ├── Terminations
│   ├── Events
│   ├── Curriculums
│   ├── Commands
│   └── Scene
│       ├── TerrainConfig
│       ├── RobotConfig
│       │   └── Actuators
│       └── <Sensors>
│
└── AgentConfig
    ├── Hyperparameters
    └── Architecture
```

With Intellisense, you can navigate through existing configurations by `Command ⌘`/`Control ^` + `click`-ing on definitions. See below clip of a user navigating through configuration definitions using this behavior:

<p align="center">
  <img src="media/config-navigation.gif" alt="Navigating Configs using Intellisense" width=600px>
</p>

We are in the process of documenting our research code. Please reach out about confusing behaviors.

## Starting a Run

In the CLI, we use [Hydra](https://hydra.cc/) to parse and override configs. All settings are exposed to hydra on execution. This allows us to conveniently change params without impacting source code and configs on CI.

### Examples

- Disable logging to `wandb`

    ```bash
    python scripts/train_rl.py --headless train.log.no_wandb=True
    ```


- Enable `test_mode` (disable logging, checkpointing, and `wandb`):

    ```bash
    python scripts/train_rl.py --headless train.log.test_mode=True
    ```

- Change reward weight for [`side_slip`](https://github.com/UWRobotLearning/WheeledLab/blob/98086f3ebded818562f739c77fd8ce26eda697aa/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env_cfg.py#L246):

    ```bash
    python scripts/train_rl.py --headless env.rewards.side_slip.weight=100.0 -r RSS_DRIFT_CONFIG
    ```

<details>
  <summary> Multirun </summary>

  Hydra also enables executing multiple runs across a list of parameters as in

```bash
python scripts/train_rl.py --headless --multirun env.rewards.side_slip.weight=10.0,20.0,100.0 -r RSS_DRIFT_CONFIG
```

However, this remains incompatible with IsaacSim as long as its `SimulationContext` only permits a single simulation to be opened per runtime execution.

</details>

## Creating a Run

Similar to `gym.register`, Wheeled Lab registers run configs through `wheeledlab_rl.utils.hydra.register_run_to_hydra`.

For instance, the set of run configurations used for the writeup and website can be found in `wheeledlab_rl.utils.configs.runs` [here](../wheeledlab_rl/configs/runs/rss_cfgs.py). They are registered in the `runs` module initialization.

You can create your own run by following these examples and registering it to the hydra registry and giving it as an argument using the "-r" flag as in:

```
python scripts/train_rl.py --headless -r <YOUR_CONFIG_NAME>
```
