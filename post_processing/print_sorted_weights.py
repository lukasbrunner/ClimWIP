#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-11-30 14:05:07 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Takes files which contain weighting and plots model names from
the highest to the lowest weighting.

"""
import os
import argparse
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    dest='filenames', nargs='*', type=str,
    help='filenames relative to base_path')
parser.add_argument(
    '--base-path', '-p', dest='base_path', type=str,
    default='/net/h2o/climphys/lukbrunn/Data/ModelWeighting/processed_CMIP5_data',
    help='Base path of the files')
parser.add_argument(
    '--best-worst', '-bw', dest='best_worst', type=int, default=None,
    help='Only show the given number of the best and worst models')
parser.add_argument(
    '--relative', '-r', dest='relative', action='store_true',
    help='Plot how often a model is in a given percentile')

args = parser.parse_args()
filenames = [os.path.join(args.base_path, fn) for fn in args.filenames]
filenames = [fn if fn.endswith('.nc') else '{}.nc'.format(fn) for fn in filenames]

if not args.relative:
    lines = None
    for filename in filenames:
        ds = xr.open_dataset(filename)
        sorter = np.argsort(ds['weights'].data)[::-1]
        if lines is None:
            config = ds.attrs['config']
            max_ = np.max([len(config)] + [len(line) for line in ds['model_ensemble'].data])
            lines = ['{config: <{max_}} |'.format(config=config, max_=max_), '-'*(max_+2)]
            lines += ['{line: <{max_}} |'.format(line=line, max_=max_)
                      for line in ds['model_ensemble'].data[sorter]]
        else:
            config = ds.attrs['config']
            lines[0] += ' {config: <{max_}} |'.format(config=config, max_=max_)
            lines[1] += '-'*(max_+3)
            for idx, line in enumerate(ds['model_ensemble'].data[sorter]):
                lines[idx+2] += ' {line: <{max_}} |'.format(line=line, max_=max_)

    if args.best_worst is not None:
        lines = lines[:args.best_worst] + ['.'] + lines[-args.best_worst:]

    for line in lines:
        print(line)

else:
    models = xr.open_dataset(filenames[0])['model_ensemble'].data
    percentiles = [33, 66]
    in_percentile = np.empty((len(filenames), len(percentiles)+1, len(models)+5)) * np.nan
    configs = []
    for i_f, filename in enumerate(filenames):
        ds = xr.open_dataset(filename)
        weights = ds['weights'].data
        configs.append(ds.attrs['config'])

        in_percentile[i_f, 0, :len(weights)] = weights < np.percentile(weights, percentiles[0])
        for i_p, percentile in enumerate(percentiles):
            try:
                in_percentile[i_f, i_p+1, :len(weights)] = (
                    (weights >= np.percentile(weights, percentile)) *
                    (weights < np.percentile(weights, percentiles[i_p+1])))
            except IndexError:
                in_percentile[i_f, i_p+1, :len(weights)] = weights >= np.percentile(weights, percentile)

    in_percentile = np.nansum(in_percentile, axis=0) / len(configs)  # mean over configs
    in_percentile = in_percentile[::-1]  # highest percentile first
    in_percentile = in_percentile.swapaxes(0, 1)  # loop over models not percentiles
    max1 = 6
    max2 = np.max([len(mm) for mm in models]) + 1

    # Some description lines
    lines = ['Split weights in mutually exclusive cathegories']
    lines.append('')
    lines.append('Included configs: ' +
                 ', '.join(['{config: <{max1}}'.format(config=config, max1=max1) for config in configs]))
    lines.append('**: always in best; --: always in worst; *: better than expected; -: worse than expected')
    lines.append('')
    lines.append('{mm: <{max2}}| '.format(mm='Models \ Percentiles', max2=max2) +
                 '>={p: <{max1}} | '.format(p=percentiles[-1], max1=max1-2) +
                 ' | '.join(['<{p: <{max1}}'.format(p=p, max1=max1-1) for p in percentiles[::-1]]))
    lines.append('-'*(max2 + (max1*(len(percentiles)+2))))

    # for each model print all categories (defined by percentiles)
    for mm, pp in zip(models, in_percentile):
        if pp[0] == 1:  # highlight models which are always good...
            mm = '*' + mm
        elif pp[-1] == 1:  # ...or bad
            mm = '-' + mm

        if pp[0] > (100 - percentiles[-1]) / 100. and pp[-1] > percentiles[0] / 100.:
            mm = '~' + mm
        elif pp[0] > (100 - percentiles[-1]) / 100.:
            mm = '*' + mm
        elif pp[-1] > percentiles[0] / 100.:
            mm = '-' + mm

        lines.append('{mm: <{max2}}| '.format(mm=mm, max2=max2) +
                     ' | '.join(['{p: .3f}'.format(p=p)
                                 for p in pp]))

    for line in lines:
        print(line)
