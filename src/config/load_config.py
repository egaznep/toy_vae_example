import json

def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)

    return cfg