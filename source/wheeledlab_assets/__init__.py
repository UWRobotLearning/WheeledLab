# source/wheeledlab_assets/__init__.py
import os

# Path to the extension source directory (this directory)
WHEELEDLAB_ASSETS_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the extension data directory
WHEELEDLAB_ASSETS_DATA_DIR = os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "data")

# Metadata and version can be handled by pyproject.toml for the Python package.
# If this __init__.py is also used by Isaac Sim extension mechanisms that expect
# extension.toml to be parsed here, then the toml import and parsing can be kept.
# For now, let's assume Python packaging is the priority.
#
# import toml
# WHEELEDLAB_ASSETS_METADATA = toml.load(os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
# __version__ = WHEELEDLAB_ASSETS_METADATA["package"]["version"]

# For simplicity, if __version__ is needed and not coming from pyproject.toml context:
__version__ = (
    "0.1.0"  # Or fetch from pyproject.toml if a more robust way is needed at runtime
)

# Import all symbols from the submodules (which will be moved to this level)
# These lines will work after mushr.py, f1tenth.py etc. are moved to this directory.
from .mushr import *
from .f1tenth import *
from .hound import *

# It's good practice to define __all__ if using 'import *'
# This lists what 'from wheeledlab_assets import *' would import.
# For now, specific important configs are listed.
# You might want to automatically populate this or list all imported names.
__all__ = [
    # Add names from mushr.py, f1tenth.py, hound.py that should be exported
    # For example, if MUSHR_SUS_2WD_CFG is in mushr.py:
    "MUSHR_CFG",
    "MUSHR_SUS_CFG",
    "MUSHR_SUS_2WD_CFG",  # From mushr.py
    "F1TENTH_CFG",  # From f1tenth.py
    # Add other HOUND configurations if they are meant to be public
    "WHEELEDLAB_ASSETS_EXT_DIR",
    "WHEELEDLAB_ASSETS_DATA_DIR",
    "__version__",
]
