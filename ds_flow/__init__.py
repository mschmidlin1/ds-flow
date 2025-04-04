"""
DS Flow - A library for data science workflows.
"""

__version__ = "0.0.0"

# Import submodules for easier access
from ds_flow import torch_flow
from ds_flow import sk_flow
from ds_flow import pandas_flow
from ds_flow import io_flow
from ds_flow import np_flow



from pathlib import Path
import importlib
import inspect
import sys
import typing

if typing.TYPE_CHECKING:
    # Static imports for type checkers
    from .general_utils import *

# Automatically discover all .py files in the current directory
current_dir = Path(__file__).parent
module_files = [f.stem for f in current_dir.glob("*.py") 
               if f.stem != "__init__" and not f.stem.startswith(".")]

# Import all modules and their contents
for module_name in module_files:
    module = importlib.import_module(f".{module_name}", __package__)
    # Copy all module attributes to the current namespace
    for name, obj in inspect.getmembers(module):
        globals()[name] = obj
        
# Create __all__ from all imported items
__all__ = list(globals().keys())
# Remove standard items that shouldn't be in __all__
for item in ['Path', 'importlib', 'inspect', 'sys', 'typing', 'TYPE_CHECKING',
             'current_dir', 'module_files', 'module_name', 'module', 'name', 'obj']:
    if item in __all__:
        __all__.remove(item) 
