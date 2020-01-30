#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: A convenience function which calls model_weighting with all
[sections] from a given configuration file (excluding [DEFAULT]).
"""
import os
import argparse
from datetime import datetime
from configparser import ConfigParser

now = datetime.today().strftime('%Y%m%d_%H%M%S')

parser = argparse.ArgumentParser()
parser.add_argument(dest='filename', help='Relative path of the config file.')
parser.add_argument('-p', dest='paper', action='store_true',
                    help='Save to a dedicated folder for easier tracking')
args = parser.parse_args()
config = ConfigParser()
config.read(args.filename)

config_name = os.path.basename(args.filename)
logname = config_name.replace('.ini', f'_{now}.log')

if args.paper:
    logname = os.path.join('logfiles_paper', logname)
else:
    logname = os.path.join('logfiles', logname)

with open(logname, 'w') as logfile:
    logfile.write('=' * 79)
    logfile.write('\n{0} {1} {0}\n'.format('=' * 5, config_name))
    logfile.write('=' * 79)

for section in config.sections():
    with open(logname, 'a') as logfile:
        logfile.write('\n\n')
        logfile.write('-' * 79)
        logfile.write('\n{0} {1} {0}\n'.format('-' * 5, section))
        logfile.write('-' * 79)
        logfile.write('\n')
    os.system(f'nice python model_weighting_main.py {section} -f {args.filename} -log-file {logname}')
