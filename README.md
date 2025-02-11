# WheeledLab
Environments, assets, workflow for open-source mobile robotics, integrated with IsaacLab.

## Installing IsaacLab
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Installing WheeledLab

```bash
# Activate the conda environment that was created via the IsaacLab setup.
conda activate <your IsaacLab env here>

git clone git@github.com:UWRobotLearning/WheeledLab.git
cd WheeledLab/source
pip install -e wheeledlab
pip install -e wheeledlab_tasks
pip install -e wheeledlab_assets
pip install -e wheeledlab_rl
```

## Training Quick Start

To start a drifting run:

```
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_DRIFT_CONFIG 
```

To start a elevation run:

```
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_ELEV_CONFIG 
```

To start a visual run:

```
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_VISUAL_CONFIG 
```

See training details in the `wheeledlab_rl` [README.md](source/wheeledlab_rl/docs/README.md)

## Setting Up VSCode

It is a million times harder to develop in IsaacLab without VSCode's Intellisense. Setting up the vscode workspace is
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
