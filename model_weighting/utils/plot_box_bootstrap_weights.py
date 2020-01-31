#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Plot a distribution of weights per model. This is intended for
the use on bootstrapped model variants.
"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from boxplot import boxplot

SAVEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/boxplots_weights'
os.makedirs(SAVEPATH, exist_ok=True)


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
    nn = ds['weights_q']
    dd = ds['weights_i']
    ds = (nn/dd).to_dataset(name='weights')
    # ds = ds['weights'].to_dataset(name='weights')
    model_ensemble = ds['model_ensemble'].data
    models = [mm.split('_')[0] for mm in model_ensemble]

    _, idxs, counts = np.unique(models, return_counts=True, return_index=True)
    for idx, count in zip(idxs, counts):
        for idx_variant in range(count):
            models[idx+idx_variant] = models[idx+idx_variant] + f'_{idx_variant}'

    ds['model'] = xr.DataArray(models, dims='model_ensemble')
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
        data = ds.sel(model=model)

        # --- variant 1 ---
        # pass

        # --- variant 2 ---
        # scale to x2, /2 etc. like in Brunner et al. 2019
        # data /= np.median(ds.median('realization').data)
        # data[data < 1.] = 1 - 1/data[data < 1.]
        # data[data >= 1] -= 1

        # --- variant 3 ---
        # normalizer = ds.sel(model=model).median('realization').data
        # data /= normalizer
        nr = len(np.unique(model_ensemble.sel(model=model).data))

        boxplot(
            ax, idx,
            data=data,
            box_quantiles=(0, 1),
            whis=None,
            width=.6,
            color=sns.xkcd_rgb['greyish'] if nr == 1 else 'blue',
            alpha=1,
        )
        xticklabels.append(f'{model} ({nr})')

    # ax.axhline(1./len(models), ls='--', color='k')
    ax.set_xticks(range(len(ds['model'])))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize='x-small')

    # --- variant 1 ---
    ax.set_ylabel('Not normalized weights (1)')

    # --- variant 2 ---
    # ax.set_ylim(-5, 5)
    # ax.set_yticks(np.arange(-5, 6, 1))
    # ax.set_yticklabels(['/6', '/5', '/4', '/3', '/2', '1', 'x2', 'x3', 'x4', 'x5', 'x6'])
    # # ax.set_yticklabels(['0.25', '0.33', '0.5', '1', '2', '3', '4'])
    # ax.set_ylabel('Model weight relative to the median')
    # ---

    ax.grid(axis='y')

    if args.title is not None:
        plt.title(args.title)

    if args.savename is None:
        plt.show()
    else:
        plt.savefig(os.path.join(SAVEPATH, args.savename), dpi=300)


if __name__ == '__main__':
    main()
