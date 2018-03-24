import click
import os

def warn(msg):
    click.secho(msg, fg='yellow') #Colors do not work in Jupyter ntb

def error(msg):
    click.secho(msg, fg='red') #Colors do not work in Jupyter ntb

def create_dir_path(path):
    # print("Created directories for path '{}' if needed.".format(path))
    return os.makedirs(path, exist_ok=True)