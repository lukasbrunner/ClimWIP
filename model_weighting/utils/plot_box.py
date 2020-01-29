#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-01-20 10:05:08 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

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
        dest='filenames', nargs='+', type=str,
        help='')
    parser.add_argument(
        '--path', '-p', dest='path', type=str, default='',
        help='')
    parser.add_argument(
        '--no-unweighted', '-u', dest='unweighted', action='store_false',
        help='')
    parser.add_argument(
        '--title', '-t', dest='title', type=str, default=None,
        help='')
    parser.add_argument(
        '--labels', '-l', dest='labels', default=None,
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--savename', '-s', dest='savename', type=str, default=None,
        help='')
    parser.add_argument(
        '--ylim', dest='ylim', default=None,
        type=lambda x: x.split(', '),
        help='')
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.filenames):
        logmsg = '--labels needs to have same length as filenames! Falling back to default'
        args.labels = None
        print(logmsg)
    return args


def read_data(filename, path):
    ds = xr.open_dataset(os.path.join(path, filename))
    ds = area_weighted_mean(ds)

    # if cfg.target_startyear_ref is not None and cfg.target_diagnostic == 'pr':
    #         ds.data /= area_weighted_mean(xr.open_dataset(
    #             os.path.join(load_path, fn))[cfg.target_diagnostic]).data

    return ds


def main():
    args = read_input()

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, right=.99, bottom=.22, top=.91)

    xticks = []
    xticklabels = []

    for xx, filename in enumerate(args.filenames):
        if filename == '':
            continue

        ds = read_data(filename, args.path)

        varn = ds.attrs['target']
        xticks.append(xx)
        xticklabels.append(ds.attrs['config'])

        if args.unweighted:
            h1 = boxplot(
                ax, xx,
                median=ds[varn],
                mean=ds[varn],
                box=ds[varn],
                whis=ds[varn],
                # whis_quantiles=(.1, .9),
                width=.8,
                color=sns.xkcd_rgb['greyish'],
                alpha=.3,
                # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
                # whis_kwargs={'caps_width': .6},
            )

        h2 = boxplot(
            ax, xx,
            median=ds[varn],
            mean=ds[varn],
            box=ds[varn],
            whis=ds[varn],
            weights=ds['weights'],
            # whis_quantiles=(.1, .9),
            width=.6,
            color=sns.xkcd_rgb['greyish'],
            alpha=1,
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # whis_kwargs={'caps_width': .6},
        )

    ax.set_xticks(xticks)
    if args.labels is not None:
        ax.set_xticklabels(args.labels, rotation=30, ha='right')
    else:
        ax.set_xticklabels(xticklabels, rotation=30, ha='right')

    try:
        unit = f' ({ds[varn].attrs["units"]})'
    except KeyError:
        unit = ''
    ax.set_ylabel(f'{varn}{unit}')
    ax.grid(axis='y')

    if args.ylim is not None:
        ax.set_ylim((float(args.ylim[0]), float(args.ylim[1])))

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
