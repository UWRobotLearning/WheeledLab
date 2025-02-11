# WheeledLab RL

A single RL run contains an overwhelming number of parameters. It is highly advised to read the configuration hierarchy through an IDE that can navigate through class definitions. 

## Creating a Run

Similar to `gym.register`, Wheeled Lab registers run configs through `wheeledlab_rl.utils.hydra.register_run_to_hydra`.

For instance, the set of run configurations used for the writeup and website can be found in `wheeledlab_rl.utils.configs.runs` [here](../wheeledlab_rl/configs/runs/rss_cfgs.py). They are registered in the `runs` module initialization.

You can create your own run by following these examples and registering it to the hydra registry and giving it as an argument using the "-r" flag as in:

```
python scripts/train_rl.py --headless -r <YOUR_CONFIG_NAME>
```
