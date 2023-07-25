import configparser
import argparse

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--config', type=str)

args = parser.parse_known_args()
config = configparser.ConfigParser()
config.read(args.config)
