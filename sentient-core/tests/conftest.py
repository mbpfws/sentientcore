import sys
import os

# Add the project root directory to the Python path
# This allows pytest to discover and import modules from the 'core' and 'graphs' packages
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root) 