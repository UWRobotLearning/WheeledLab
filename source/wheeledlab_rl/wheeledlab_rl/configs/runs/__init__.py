from .rss_cfgs import *
from .f1tenth_cfgs import *

from wheeledlab_rl.utils.hydra import register_run_to_hydra

register_run_to_hydra("RSS_DRIFT_CONFIG", RSS_DRIFT_CONFIG)
register_run_to_hydra("RSS_ELEV_CONFIG", RSS_ELEV_CONFIG)
register_run_to_hydra("RSS_VISUAL_CONFIG", RSS_VISUAL_CONFIG)

register_run_to_hydra("F1TENTH_DRIFT_CONFIG", F1TENTH_DRIFT_CONFIG)