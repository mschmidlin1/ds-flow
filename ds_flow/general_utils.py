





import os
import sys


def get_venv_name():
    # Check if the script is running inside a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Get the path to the virtual environment and extract its name
        venv_path = sys.prefix
        venv_name = os.path.basename(venv_path)
        return venv_name
    else:
        return None