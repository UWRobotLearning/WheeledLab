<img src="docs/media/fig1.png" alt="fig1" />

---

# WheeledLab

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.0-silver.svg)](https://isaac-sim.github.io/IsaacLab/v2.0.0/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

Environments, assets, workflow for open-source mobile robotics, integrated with IsaacLab.

[Website](https://uwrobotlearning.github.io/WheeledLab/) | [Paper](https://arxiv.org/abs/2502.07380)

## Installing IsaacLab (~30 min)

WheeledLab is built atop Isaac Lab. If you do not yet have Isaac Lab installed, it is open-source and installation instructions for Isaac Sim v4.5.0 and Isaac Lab v2.0.2 can be found below:

```bash
# Create a conda environment named env_isaaclab and install Isaac Sim v4.5.0 in it:
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # Or `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118` for CUDA 11
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab v2.0.2 (make sure you have build dependencies first, e.g. `sudo apt install cmake build-essential` on ubuntu)
git clone --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git
./isaaclab.sh -i

# Downgrade gymnasium to <v1.0
pip install gymnasium==0.29.1
```

Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

If you already have IsaacLab you can skip this and instead head [here](#create-new-isaaclab-conda-environment) to set up a new conda environment for this repository.

### Create New IsaacLab Conda Environment

We recommend setting up a new [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) environment to include both `IsaacLab` packages and `WheeledLab` packages. You can do this using Isaac Lab's convenient setup scripts:

```bash
cd <IsaacLab>
./isaaclab.sh --conda WL
conda activate WL
./isaaclab.sh -i
```

## Installing WheeledLab (~5 min)

```bash
# Activate the conda environment that was created via the IsaacLab setup.
conda activate <your IsaacLab env here> # 'WL' if you followed instructions above

git clone git@github.com:UWRobotLearning/WheeledLab.git
cd WheeledLab/source
pip install -e wheeledlab
pip install -e wheeledlab_tasks
pip install -e wheeledlab_assets
pip install -e wheeledlab_rl
```

## Training Quick Start

Training runs can take a couple hours to produce a transferable policy.

To start a drifting run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_DRIFT_CONFIG
```

To start a elevation run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_ELEV_CONFIG
```

To start a visual run:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_VISUAL_CONFIG
```

Though optional (and free), we strongly advise using [Weights & Biases](https://wandb.ai/site/) (`wandb`) to record and track training status. Logging to `wandb` is turned on by default. If you would like to disable it, add `train.log.no_wandb=True` to the CLI arguments.

See more details about training in the `wheeledlab_rl` [README.md](source/wheeledlab_rl/docs/README.md)

## Deployment

A separate repository is maintained for existing integrations and deployments. See https://github.com/UWRobotLearning/RealLab for code.

### Current Integrations

1. HOUND [1] - https://github.com/UWRobotLearning/RealLab/tree/hound
2. MuSHR [2] - https://github.com/UWRobotLearning/RealLab/tree/mushr
3. F1Tenth [3] - (coming soon)

If you have an integration or request for a platform not seen above, please contact us or consider contributing!

## Setting Up VSCode

It is a million times harder to develop in IsaacLab without Intellisense. Setting up the vscode workspace is
STRONGLY advised.

0. Find where your `IsaacLab` directory currently is. We'll refer to it as `<IsaacLab>` in this section. Move the VSCode tools to this workspace.

   ```bash
   cd <WheeledLab>
   cp -r <IsaacLab>/.vscode/tools ./.vscode/
   cp -r <IsaacLab>/.vscode/*.json ./.vscode/
   ```

1. Change `.vscode/tasks.json` line 11

   ```json
   "command": "${workspaceFolder}/../IsaacLab/isaaclab.sh -p ${workspaceFolder}/.vscode/tools/setup_vscode.py"
   ```

   to

   ```json
   "command": "<IsaacLabDir>/isaaclab.sh -p ${workspaceFolder}/.vscode/tools/setup_vscode.py"
   ```

2. `Ctrl` + `Shift` + `P` to bring up the VSCode command palette. type `Tasks:Run Task` or type until you see it show up and highlight it and press `Enter`.
3. Click on `setup_python_env`. Follow the prompts until you're able to run the task. You should see a console at the bottom and the status of the task.
4. If successful, you should now have `.vscode/{settings.json, launch.json}` in your `<WheeledLab>` repo and `settings.json` should have a populated list of paths under the `"python.analysis.extraPaths"` key.

### If it still doesn't work

The `setup_vscode` task doesn't work for me for whatever reason. If that's true for you too, add the following lines to the end of the list under the key `"python.analysis.extraPaths"` in the `.vscode/settings.json` file:

```json
    "<IsaacLab>/source/isaaclab",
    "<IsaacLab>/source/isaaclab_assets",
    "<IsaacLab>/source/isaaclab_tasks",
    "<IsaacLab>/source/isaaclab_rl",
```

## References

### This work

```
@misc{2502.07380,
Author = {Tyler Han and Preet Shah and Sidharth Rajagopal and Yanda Bao and Sanghun Jung and Sidharth Talia and Gabriel Guo and Bryan Xu and Bhaumik Mehta and Emma Romig and Rosario Scalise and Byron Boots},
Title = {Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics},
Year = {2025},
Eprint = {arXiv:2502.07380},
}
```

### Cited

[1] Sidharth Talia, Matt Schmittle, Alexander Lambert, Alexander Spitzer, Christoforos Mavrogiannis, and Siddhartha S. Srinivasa.Demonstrating HOUND: A Low-cost Research Platform for High-speed Off-road Underactuated Nonholonomic Driving, July 2024.URL http://arxiv.org/abs/2311.11199.arXiv:2311.11199 [cs].

[2] Siddhartha S. Srinivasa, Patrick Lancaster, Johan Michalove, Matt Schmittle, Colin Summers, Matthew Rockett, Rosario Scalise, Joshua R. Smith, Sanjiban Choudhury, Christoforos Mavrogiannis, and Fereshteh Sadeghi.MuSHR: A Low-Cost, Open-Source Robotic Racecar for Education and Research, December 2023.URL http://arxiv.org/abs/1908.08031.arXiv:1908.08031 [cs].

[3] Matthew O’Kelly, Hongrui Zheng, Dhruv Karthik, and Rahul Mangharam. F1TENTH: An Open-source Eval- uation Environment for Continuous Control and Reinforcement Learning. In Proceedings of the NeurIPS 2019 Competition and Demonstration Track, pages 77– 89. PMLR, August 2020. URL https://proceedings.mlr. press/v123/o-kelly20a.html. ISSN: 2640-3498.
