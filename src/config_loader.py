import os
import yaml

def load_config():

    # Setting current path
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Current Folder
    config_path = os.path.join(current_dir, "..", "config", "config.yaml") # Config folder
    
    # opening the config yaml file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)