#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-01-29 14:37:37 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Plot boxes with weights distribution over all given pseudo observations.
"""
import os
import argparse
import warnings
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from boxplot import boxplot

warnings.filterwarnings('ignore')

period_ref = None  # slice('1979', '2014')

PLOTPATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../plots/box_weights_pseudo_obs/')
os.makedirs(PLOTPATH, exist_ok=True)


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filenames', type=str, nargs='+',
        help='Valid weights file (should end with .nc)')
    parser.add_argument(
        '--title', '-t', dest='title', type=str, default=None,
        help='')
    parser.add_argument(
        '--savename', '-s', dest='savename', type=str, default=None,
        help='')
    args = parser.parse_args()

    return args


def main():
    args = read_input()

    ds_list = []
    for filename in args.filenames:
        ds = xr.open_dataset(filename).squeeze()
        cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
        ds['observation'] = xr.DataArray([cfg.obs_id], dims='observation')
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='observation')

    ds = ds['weights']
    ds = ds.sortby(ds.mean('observation'), ascending=False)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, right=.88, bottom=.22, top=.91)

    models = ds['model_ensemble'].data
    xticklabels = []
    for idx, model in enumerate(models):
        boxplot(
            ax, idx,
            median=ds.sel(model_ensemble=model),
            mean=ds.sel(model_ensemble=model),
            box=ds.sel(model_ensemble=model),
            whis=ds.sel(model_ensemble=model),
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
        )
        xticklabels.append(model.rstrip('_CMIP6'))

    ax.set_xticks(range(len(ds['model_ensemble'])))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize='x-small')

    ax.set_ylabel('Weights (1)')
    ax.grid(axis='y')

    ax.axhline(1/len(models), ls='--', color='k')

    if args.title is not None:
        plt.title(args.title)

    if args.savename is None:
        plt.show()
    else:
        plt.savefig(os.path.join(PLOTPATH, args.savename), dpi=300)


if __name__ == '__main__':
    main()
