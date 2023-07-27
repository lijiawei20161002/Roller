import configparser
import argparse
import sys

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--config', type=str, default=None)

args, leftovers = parser.parse_known_args()
config = configparser.ConfigParser()
#args.config = '/Users/lijiawei/desktop/Roller/configuration/test.ini'
if args.config is not None:
    config.read(args.config)
else:
    sys.exit("Usage --config <config file path>")
