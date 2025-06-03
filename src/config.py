import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, default='./config/config.yaml')
args = parser.parse_args()

def load_config():
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config