import os
import sys
import yaml

def load_config():

    # Setting current path
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Current Folder
    project_root = os.path.abspath(os.path.join(current_dir, "..")) # Project Folder
    sys.path.insert(0, project_root) # Setting the Project Folder as a priority to find modules

    config_path = os.path.join(current_dir, "..", "config", "config.yaml") # Config folder
    # opening the config yaml file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()