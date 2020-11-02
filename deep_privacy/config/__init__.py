import argparse
from .base import Config


def default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser
