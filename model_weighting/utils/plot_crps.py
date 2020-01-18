#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-01-17 15:30:59 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Calculate the CRPS of the target variable for the future period
based on a perfect model test.

"""
import os
import warnings
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
import seaborn as sns

from utils_python.xarray import area_weighted_mean

from boxplot import boxplot


SAVEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/skill'
os.makedirs(SAVEPATH, exist_ok=True)


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
        '--title', '-t', dest='title', type=str, default=None,
        help='')
    parser.add_argument(
        '--labels', '-l', dest='labels', default=None,
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--exclude-ensembles', '-e', dest='exclude_ensembles', default=None,
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--savename', '-s', dest='savename', type=str, default=None,
        help='')
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.filenames):
        logmsg = '--labels needs to have same length as filenames! Falling back to default'
        args.labels = None
        print(logmsg)

    if args.exclude_ensembles is None:
        args.exclude_ensembles = ['none'] * len(args.filenames)

    return args


def crps_data(data, weights, model_ensemble=None, exclude_ensembles='all'):
    """Calculate the CRPS.

    Parameters
    ----------
    data : np.array, shape (N,)
    weights : np.array, shape (N,)
        Exactly one element of data has to be NaN. This will be used to
        identify the index of the perfect model.
    model_ensemble : np.array, shape (N,), optional
        List of model identifiers. Only needs to be given if exclude_ensembles
        is not 'none'.
    exclude_ensembles : {'all', 'same', 'none'}, optional
        - all: only use one ensemble member per model
        - same: use all ensemble members but exclude ensemble members from
          the same model as the perfect model
        - none: use all models and members except the perfect one

    Returns
    -------
    skill : float
    """
    assert np.isnan(weights).sum() == 1, 'exactly one weight has to be np.nan!'
    assert exclude_ensembles in ['all', 'same', 'none']
    if exclude_ensembles != 'none' and model_ensemble is None:
        raise ValueError('If exclude_ensembles is not none model_ensemble has to be given!')

    idx_perfect = np.where(np.isnan(weights))[0][0]
    if exclude_ensembles != 'none':
        models = [mm.split('_')[0] for mm in model_ensemble.data]
        if exclude_ensembles == 'all':
            _, idx_sel = np.unique(models, return_index=True)
            idx_sel = idx_sel[idx_sel != idx_perfect]
        elif exclude_ensembles == 'same':
            idx_sel = np.where(np.array(models) != models[idx_perfect])
        else:
            raise ValueError
    else:
        idx_sel = np.delete(np.arange(len(data)), idx_perfect)

    data_ref = data[idx_perfect]
    data_test = data[idx_sel]
    weights_test = weights[idx_sel]

    baseline = crps_ensemble(data_ref, data_test)
    weighted = crps_ensemble(data_ref, data_test, weights=weights_test)
    return (baseline - weighted) / baseline


def crps_xarray(ds, varn, exclude_ensembles='all'):
    """
    Handle ensemble members and call CRPS calculation.

    Parameters
    ----------
    ds : xarray.Dataset
    varn : string

    Returns
    -------
    skill : xarray.DataArray
    """
    skill = xr.apply_ufunc(
        crps_data, ds[varn], ds['weights'],
        input_core_dims=[['model_ensemble'], ['model_ensemble']],
        kwargs={'model_ensemble': ds['model_ensemble'],
                'exclude_ensembles': exclude_ensembles},
        vectorize=True,
    )
    if 'lat' in skill.dims:
        skill = area_weighted_mean(skill, suppress_warning=True)
    return skill


def read_data(filename, path):
    ds = xr.open_dataset(os.path.join(path, filename))
    ds = area_weighted_mean(ds)
    return ds


def main():
    """load files and call functions"""
    args = read_input()

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, right=.99, bottom=.22, top=.91)

    xticks = []
    xticklabels = []

    for xx, ff in enumerate(args.filenames):
        if ff == '':
            continue

        ds = read_data(ff, args.path)
        varn = ds.attrs['target']
        region = ds.attrs['region']

        skill = crps_xarray(ds, varn, args.exclude_ensembles[xx])

        idx_sort = skill.argsort().data[::-1]
        print(f'{ds.attrs["config"]} {args.exclude_ensembles[xx]}')
        # print('Best CRPS:')
        # print('\n'.join([
        #     f' {ds["model_ensemble"].data[idx]}, {skill.data[idx]:.3f}' for idx in idx_sort[:3]]))
        # print('Worst CRPS:')
        # print('\n'.join([
        #     f' {ds["model_ensemble"].data[idx]}, {skill.data[idx]:.3f}' for idx in idx_sort[-3:]]))

        weights = ds['weights'].mean('perfect_model_ensemble', skipna=True)
        weights /= weights.median('model_ensemble')

        print('\n'.join([
            f' {ds["model_ensemble"].data[idx]}, {skill.data[idx]:.3f}'  # , {weights[idx].data:.3f}'
            for idx in idx_sort]))

        xticks.append(xx)
        xticklabels.append(ds.attrs['config'] + f' {args.exclude_ensembles[xx]}')

        median = skill.median('perfect_model_ensemble').data
        mean = skill.mean('perfect_model_ensemble').data
        p05, p95 = skill.quantile((.05, .95), 'perfect_model_ensemble').data

        # sorter = skill.argsort()[::-1]
        # model_ensemble = ', '.join(skill['perfect_model_ensemble'].data[sorter])
        # print(f'{model_ensemble}')

        boxplot(
            ax, xx,
            mean=skill,
            median=skill,
            box=skill,
            whis=skill,
            width=.8,
            alpha=1,
            color=sns.xkcd_rgb['greyish'],
            # whis_kwargs={'caps_width': .6},
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
        )

    ax.grid(axis='y')
    ax.axhline(0, color='k', zorder=2)

    ax.set_xticks(xticks)
    if args.labels is not None:
        ax.set_xticklabels(args.labels, rotation=30, ha='right')
    else:
        ax.set_xticklabels(xticklabels, rotation=30, ha='right')

    ax.set_ylim(-1, 1)
    ax.set_yticklabels(np.around(ax.get_yticks() * 100).astype(int))
    ax.set_ylabel('Relative CRPS change (%)', labelpad=-3)

    if args.title is not None:
        plt.title(args.title)
    else:
        plt.title(f'Change in CRPS')

    if args.savename is None:
        plt.show()
    else:
        plt.savefig(os.path.join(SAVEPATH, args.savename), dpi=300)


if __name__ == "__main__":
    main()
