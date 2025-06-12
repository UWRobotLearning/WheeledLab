"""
Boilerplate code for starting up IsaacLab backend
"""

import argparse
import sys

def startup(parser=None, prelaunch_callback=None, register_cfgs=True):
    from isaaclab.app import AppLauncher
    '''
    Startup IsaacLab backend. Imports wheeled_gym environments optionally.
    Args:
        parser: argparse.ArgumentParser, optional, default=None
            Argument parser to add arguments to.
        prelaunch(args): function to be executed right before launching the app, optional, default=None
    Returns:
        simulation_app: omni.isaac.dynamic_control.DynamicControl, omni.isaac.dynamic_control._dynamic_control.DynamicControl
            Simulation app instance.
        args_cli: argparse.Namespace
            Parsed command line arguments.
    '''

    if parser is None:
        parser = argparse.ArgumentParser(description="Used Boilerplate Starter.")

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if prelaunch_callback is not None:
        prelaunch_callback(args_cli)

    args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    if register_cfgs:
        import wheeledlab_tasks # env configs
        import wheeledlab_rl.configs.runs # run configs

    return simulation_app, args_cli