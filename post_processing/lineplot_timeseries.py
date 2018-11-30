#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-11-14 11:38:50 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import warnings
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from cdo import Cdo
from tempfile import TemporaryDirectory
cdo = Cdo()
# sns.set()

from utils_python import utils
from utils_python.xarray import area_weighted_mean
from utils_python.math import variance

from model_weighting.model_weighting import set_up_filenames, get_filenames
from model_weighting.functions.diagnostics import calculate_basic_diagnostic

REGION_DIR = '{}/../cdo_data/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'  # 'seamask_g025.nc'
SAVE_PATH = '/net/h2o/climphys/lukbrunn/Plots/ModelWeighting/lineplot_timeseries'

variance = np.vectorize(variance, signature='(n)->()', excluded=['weights', 'biased'])
period_ref = slice('1976', '2005')
perfect_model_ensemble = 'MIROC5_r1i1p1'


def read_config():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='config', nargs='?', default='DEFAULT',
        help='Name of the configuration to use (optional).')
    parser.add_argument(
        '--filename', '-f', dest='filename', default='../configs/config.ini',
        help='Relative or absolute path/filename.ini of the config file.')
    args = parser.parse_args()
    cfg = utils.read_config(args.config, args.filename)
    utils.log_parser(cfg)
    return cfg


def read_data(cfg):
    varn = cfg.target_diagnostic
    fn = set_up_filenames(cfg)
    ds_list = []
    i = 0
    for filename, model_ensemble in zip(
            *get_filenames(fn, varn, all_members=True)):

        ds = calculate_basic_diagnostic(
            filename, varn,
            outfile=None,
            time_period=None,
            season=cfg.target_season,
            time_aggregation=None,
            mask_ocean=cfg.target_masko,
            region=cfg.region)

        # xarray spams warnings about the masked vales
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            ds = ds.resample(time='1a').mean()
        ds = area_weighted_mean(ds)
        ds = ds.sel(time=slice('1950', None))
        # ds[varn] -= ds[varn].sel(time=period_ref).mean('time')
        ds['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        ds['time'].data = ds['time'].dt.year.data
        ds_list.append(ds)
        i += 1
        # if i == 10:  # DEBUG
        #     break
    return xr.concat(ds_list, dim='model_ensemble')


def read_obs(cfg):
    varn = cfg.target_diagnostic
    path = cfg.obs_path
    source = cfg.obsdata
    filename = os.path.join(path, '{}_mon_{}_g025.nc'.format(varn, source))
    if not os.path.isfile(filename):
        return None

    ds = calculate_basic_diagnostic(
        filename, varn,
        outfile=None,
        time_period=None,
        season=cfg.target_season,
        time_aggregation=None,
        mask_ocean=cfg.target_masko,
        region=cfg.region,
        regrid=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        ds = ds.resample(time='1a').mean()
    ds = area_weighted_mean(ds)
    ds = ds.sel(time=slice('1950', None))
    # ds[varn] -= ds[varn].sel(time=period_ref).mean('time')
    ds['time'].data = ds['time'].dt.year.data
    return ds


def read_weights(model_ensemble, cfg):
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds = xr.open_dataset(filename)
    ds = ds.sel(model_ensemble=model_ensemble)
    return ds['weights']


def plot(cfg):
    varn = cfg.target_diagnostic
    fig, ax = plt.subplots()#figsize=(15, 5))

    ds = read_data(cfg)
    data, xx = ds[varn].data, ds['time'].data

    l2 = None

    # --- unweighted baseline ---
    # for dd in data:
    #     ax.plot(xx, dd, color='gray', alpha=.2, lw=.5)  # all lines
    l1a = ax.fill_between(xx, data.mean(axis=0) - data.std(axis=0),  # StdDev shading
                    data.mean(axis=0) + data.std(axis=0),
                    color='gray', alpha=.2)

    # plot mean
    [l1b] = ax.plot(xx, data.mean(axis=0), color='gray', lw=2)

    # --- weighted ----
    weights = read_weights(ds['model_ensemble'].data, cfg)
    assert np.all(weights['model_ensemble'] == ds['model_ensemble'])
    if 'perfect_model_ensemble' in weights.dims:
        weights = weights.sel(perfect_model_ensemble=perfect_model_ensemble)
        model_ensemble = list(ds['model_ensemble'].data)
        model_ensemble.remove(perfect_model_ensemble)
        weights = weights.sel(model_ensemble=model_ensemble)
        data = ds.sel(model_ensemble=model_ensemble)[varn].data

        # plot 'true' model
        [l2] = ax.plot(xx, ds.sel(model_ensemble=perfect_model_ensemble)[varn].data,
                color='k', lw=2)

    weights = weights.data
    def color(ww):
        """Different colors due to weights"""
        if ww > .05:
            return 'red'
        elif ww > .01:
            return 'orange'
        elif ww > .005:
            return 'yellow'
        return 'gray'
    for dd, ww in zip(data, weights):
        ww /= weights.max()
        ax.plot(xx, dd,
                color='darkred', #color(ww),
                alpha=ww,
                lw=.5)  # all lines
    [l3c] = ax.plot([], [], color='darkred', lw=.5)
    l3a = ax.fill_between(xx,
                    (np.average(data, weights=weights, axis=0) -
                     np.sqrt(variance(data.swapaxes(0, 1), weights=weights))),
                    (np.average(data, weights=weights, axis=0) +
                     np.sqrt(variance(data.swapaxes(0, 1), weights=weights))),
                          color='darkred', alpha=.2)

    [l3b] = ax.plot(xx, np.average(data, weights=weights, axis=0), color='darkred', lw=2)

    # --- observations ---
    obs = read_obs(cfg)
    if obs is not None:
        [l2] = ax.plot(obs['time'].data, obs[varn].data, color='k', lw=2)

    ax.set_xlabel('Year')
    ax.set_xlim(1950, 2060)

    # ax.set_ylabel('Temperature ($\degree$C)')
    # ax.set_ylim(-5, 10)

    ax.grid()
    ax.set_title(varn)

    if l2 is None:
        plt.legend([(l1a, l1b), (l3a, l3b)],
                   ['Unweighted mean & StdDev', 'Weighted mean & StdDev'],
                   loc='upper left')

    else:
        plt.legend(
            [(l1a, l1b),
             (l3a, l3b),
             l2,
             #l3c
            ],
            ['Unweighted mean & StdDev',
             'Weighted mean & StdDev',
             'Observations',
             #'Models by weight'
            ],
            loc='upper left'
        )

    filename = os.path.join(SAVE_PATH, 'lines_{}'.format(cfg.config))
    plt.savefig(filename + '.png', dpi=300)
    plt.savefig(filename + '.pdf', dpi=300)
    print('Saved: {}'.format(filename + '.png'))


def main():
    cfg = read_config()
    plot(cfg)


if __name__ == '__main__':
    main()
