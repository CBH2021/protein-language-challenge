import click
import yaml

from challenge import main
from challenge.utils import setup_logging


@click.group()
def cli():
    """ CLI for the challenge """
    pass


@cli.command()
@click.option(
    '-c',
    '--config-filename',
    default=['experiments/config.yml'],
    multiple=True,
    help=(
        'Path to training configuration file. If multiple are provided, runs will be '
        'executed in order'
    )
)
@click.option('-r', '--resume', default=None, type=str, help='path to checkpoint')
def train(config_filename: str, resume: str):
    """ Entry point to start training run(s). """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        setup_logging(config)
        main.train(config, resume)


@cli.command()
@click.option('-c', '--config-filename', default=['experiments/config.yml'], help='Path to model configuration file.')
@click.option('-m', '--model_path', default='model.pth', type=str, help='Path to trained model')
@click.option('-i', '--test_path', default=None, type=str, help='Path to test data')
def eval(config_filename: str, model_path: str, test_path: str):
    config = load_config(config_filename)
    main.eval(config, model_path, test_path)


@cli.command()
@click.option('-c', '--config-filename', default='config.yml', help='Path to model configuration file.')
@click.option('-m', '--model_path', default='model.pth', type=str, help='Path to trained model')
@click.option('-i', '--data', default=None, type=str, help='Path to prediction data')
def predict(config_filename: str, model_path: str, data: str):
    config = load_config(config_filename)
    main.predict(config, model_path, data)


def load_config(filename: str) -> dict:
    """ Load a configuration file as YAML. """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config
