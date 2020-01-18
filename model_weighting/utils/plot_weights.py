#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from utils_python.xarray import area_weighted_mean
from boxplot import boxplot

SAVEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/boxplots'


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filename_pattern', type=str,
        help='')
    parser.add_argument(
        '--path', '-p', dest='path', type=str, default='',
        help='')
    parser.add_argument(
        '--title', '-t', dest='title', type=str, default=None,
        help='')
    parser.add_argument(
        '--savename', '-s', dest='savename', type=str, default=None,
        help='')
    args = parser.parse_args()
    return args


def preprocess(ds):
    ds = ds['weights'].to_dataset(name='weights')
    model_ensemble =  ds['model_ensemble'].data
    model = [mm.split('_')[0] for mm in  model_ensemble]
    ds['model'] = xr.DataArray(model, dims='model_ensemble')
    ds = ds.swap_dims({'model_ensemble': 'model'})
    return ds


def main():
    args = read_input()

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, right=.88, bottom=.22, top=.91)

    ds = xr.open_mfdataset(os.path.join(args.path, args.filename_pattern),
                           use_cftime=True,
                           concat_dim='realization',
                           combine='nested',
                           preprocess=preprocess).load()

    model_ensemble = ds['model_ensemble']

    ds = ds['weights']
    ds = ds.sortby(ds.mean('realization'), ascending=False)

    models = ds['model'].data
    xticklabels = []
    for idx, model in enumerate(models):
        boxplot(
            ax, idx,
            median=ds.sel(model=model),
            mean=ds.sel(model=model),
            box=ds.sel(model=model),
            whis=ds.sel(model=model),
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
        )
        nr = len(np.unique(model_ensemble.sel(model=model).data))
        xticklabels.append(f'{model} ({nr})')

    ax.set_xticks(range(len(ds['model'])))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize='small')

    ax.set_ylabel('Weights (1)')
    ax.grid(axis='y')

    noramlizer = ds.mean('realization').median('model').data
    yticks = ax.get_yticks()
    yticklabels = np.around(yticks / noramlizer, 2)

    ax2 = ax.twinx()
    ax.set_ylim(0, yticks[-1])
    ax2.set_yticks(yticks)
    ax2.set_ylim(0, yticks[-1])
    ax2.set_yticklabels(yticklabels)
    ax2.set_ylabel('Weights scaled by median weight (1)')

    if args.title is not None:
        plt.title(args.title)

    if args.savename is None:
        plt.show()
    else:
        plt.savefig(os.path.join(SAVEPATH, args.savename), dpi=300)



if __name__ == '__main__':
    main()
