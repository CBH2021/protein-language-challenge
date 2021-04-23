import argparse
from challenge import main
from challenge.cli import load_config

# read inputs provided by user
parser = argparse.ArgumentParser()
parser.add_argument('--data', dest="data")
args = parser.parse_args()

config = load_config("config.yml")
main.predict(config, "model.pth", args.data)


