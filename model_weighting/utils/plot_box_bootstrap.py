#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Plots multiple boxes based on a given filename pattern. Indented
for plotting the output of bootstrapping model variants.

"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from utils_python.xarray import area_weighted_mean
from boxplot import boxplot, quantile

SAVEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/boxplots_bootstrap'


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
        '--no-unweighted', '-u', dest='unweighted', action='store_false',
        help='')
    parser.add_argument(
        '--no-mean', '-m', dest='mean_', action='store_false',
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
    ds = ds.drop_dims('perfect_model_ensemble')
    model_ensemble = ds['model_ensemble'].data
    models = [mm.split('_')[0] for mm in model_ensemble]
    _, idxs, counts = np.unique(models, return_counts=True, return_index=True)
    for idx, count in zip(idxs, counts):
        for idx_variant in range(count):
            models[idx+idx_variant] = models[idx+idx_variant] + f'_{idx_variant}'

    ds['model'] = xr.DataArray(models, dims='model_ensemble')
    ds = ds.swap_dims({'model_ensemble': 'model'})
    ds = ds.drop_vars('model_ensemble')
    ds = area_weighted_mean(ds)
    return ds


def main():
    args = read_input()

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(left=.1, right=.88, bottom=.22, top=.91)

    ds = xr.open_mfdataset(os.path.join(args.path, args.filename_pattern),
                           use_cftime=True,
                           concat_dim='realization',
                           combine='nested',
                           preprocess=preprocess).load()
    varn = ds.attrs['target']

    percentiles = []
    percentiles_w = []
    for idx in ds['realization'].data:
        ds_sel = ds.sel(realization=idx)
        percentiles.append(list(
            quantile(ds_sel[varn].data, (.1, .25, .5, .75, .9))) + [np.average(ds_sel[varn].data)])
        percentiles_w.append(list(
            quantile(ds_sel[varn].data, (.1, .25, .5, .75, .9), ds_sel['weights'].data)) + [
                np.average(ds_sel[varn].data, weights=ds_sel['weights'].data)])

        if args.unweighted:
            h1 = boxplot(
                ax, idx,
                median=ds_sel[varn],
                mean=ds_sel[varn],
                box=ds_sel[varn],
                whis=quantile(ds_sel[varn].data, (.1, .9)),
                width=.8,
                color=sns.xkcd_rgb['greyish'],
                alpha=.3,
                # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # whis_kwargs={'caps_width': .6},
            )

        h2 = boxplot(
            ax, idx,
            median=ds_sel[varn],
            mean=ds_sel[varn],
            box=ds_sel[varn],
            whis=quantile(ds_sel[varn].data, (.1, .9), ds_sel['weights']),
            weights=ds_sel['weights'],
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # whis_kwargs={'caps_width': .6},
        )

    if args.mean_:
        # plot the mean of the percentiles
        percentiles = np.mean(percentiles, axis=0)
        percentiles_w = np.mean(percentiles_w, axis=0)

        if args.unweighted:
            h1 = boxplot(
                ax, idx+2,
                median=percentiles[2],
                mean=percentiles[-1],
                box=percentiles[np.array([1, 3])],
                whis=percentiles[np.array([0, 4])],
                width=.8,
                color=sns.xkcd_rgb['greyish'],
                alpha=.3,
                # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # whis_kwargs={'caps_width': .6},
            )

        h2 = boxplot(
            ax, idx+2,
            median=percentiles_w[2],
            mean=percentiles_w[-1],
            box=percentiles_w[np.array([1, 3])],
            whis=percentiles_w[np.array([0, 4])],
            weights=ds_sel['weights'],
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # whis_kwargs={'caps_width': .6},
        )

        # plot the percentiles of the mean
        ds_mean = ds.mean('realization')
        if args.unweighted:
            h1 = boxplot(
                ax, idx+3,
                median=ds_mean[varn],
                mean=ds_mean[varn],
                box=ds_mean[varn],
                whis=quantile(ds_mean[varn].data, (.1, .9)),
                width=.8,
                color=sns.xkcd_rgb['greyish'],
                alpha=.3,
                # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # whis_kwargs={'caps_width': .6},
            )

        h2 = boxplot(
            ax, idx+3,
            median=ds_mean[varn],
            mean=ds_mean[varn],
            box=ds_mean[varn],
            whis=quantile(ds_mean[varn].data, (.1, .9), ds_mean['weights']),
            weights=ds_mean['weights'],
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # whis_kwargs={'caps_width': .6},
        )

    xticks = [*range(len(ds['realization']))]
    # xticklabels = xticks
    xticklabels = [''] * len(xticks)
    xticks = xticks + [len(ds['realization']) + 1, len(ds['realization']) + 2]
    xticklabels = xticklabels + ['P-mean', 'W-mean']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha='right')
    # ax.set_ylim(2, 6)

    try:
        unit = f' ({ds[varn].attrs["units"]})'
    except KeyError:
        unit = ''
    ax.set_ylabel(f'{varn}{unit}')
    ax.grid(axis='y')

    if args.unweighted:
        plt.legend((h1, h2), ('unweighted', 'weighted'))

    if args.title is not None:
        plt.title(args.title)

    if args.savename is None:
        plt.show()
    else:
        plt.savefig(os.path.join(SAVEPATH, args.savename), dpi=300)


if __name__ == '__main__':
    main()
