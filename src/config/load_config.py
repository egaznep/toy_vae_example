import yaml
import time

def load_config(path):
    with open(path, 'r') as f:
        # load from disk
        cfg = yaml.safe_load(f)
        # process formatting if "print_handlers exist"
        if 'print_handlers' in cfg:
            cfg['print_handlers']['date'] = time.strftime("%Y%m%d-%H%M")
            traverse_dict(cfg, cfg['print_handlers'])
    return cfg

def traverse_dict(cfg, print_handlers):
    if isinstance(cfg, dict):
        for key in cfg:
            cfg[key] = traverse_dict(cfg[key], print_handlers)
        return cfg
    elif isinstance(cfg, str):
        return cfg.format(**print_handlers)
    else:
        return cfg