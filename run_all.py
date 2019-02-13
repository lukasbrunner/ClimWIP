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

parser = argparse.ArgumentParser()
parser.add_argument(dest='filename', help='Relative path of the config file.')
args = parser.parse_args()
config = ConfigParser()
config.read(args.filename)

logname = os.path.basename(args.filename)
with open('model_weighting.log', 'w') as logfile:
    logfile.write('=' * 79)
    logfile.write('\n{0} {1} {0}\n'.format('=' * 5, logname))
    logfile.write('=' * 79)

for section in config.sections():
    os.makedirs(config[section]['save_path'], exist_ok=True)
    os.makedirs(config[section]['plot_path'], exist_ok=True)
    os.makedirs(config[section]['plot_path'].replace('process_plots', 'timeseries'), exist_ok=True)
    with open('model_weighting.log', 'a') as logfile:
        logfile.write('\n\n')
        logfile.write('-' * 79)
        logfile.write('\n{0} {1} {0}\n'.format('-' * 5, section))
        logfile.write('-' * 79)
        logfile.write('\n')
    os.system(f'python model_weighting.py {section} -f {args.filename} -log-file model_weighting.log')
    os.system(f'python post_processing/lineplot_timeseries.py {section} -f ../{args.filename}')

now = datetime.today().strftime('%Y%m%d_%H%M%S')
logname = logname.replace('.ini', f'_{now}.log')
os.rename('model_weighting.log', f'logfiles/{logname}')
