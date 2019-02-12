#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-01-29 10:42:25 lukbrunn>

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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()

from utils_python import utils
from utils_python.xarray import area_weighted_mean
from utils_python.math import variance, quantile2

from model_weighting.model_weighting import set_up_filenames, get_filenames
from model_weighting.functions.diagnostics import calculate_basic_diagnostic

REGION_DIR = '{}/../cdo_data/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'  # 'seamask_g025.nc'

variance = np.vectorize(variance, signature='(n)->()', excluded=['weights', 'biased'])
period_ref = slice('1976', '2005')
perfect_model_ensemble = 'MIROC5_r1i1p1'
REGRID_OBS = [
    'ERA-Interim',
]


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
    parser.add_argument(
        '--change', dest='change', action='store_true',
        help='Plot change instead of absolute values')
    args = parser.parse_args()
    cfg = utils.read_config(args.config, args.filename)
    utils.log_parser(cfg)
    return cfg, args


def read_data(cfg, change):
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
        if change:
            ds[varn] -= ds[varn].sel(time=slice(
                str(cfg.target_startyear_ref),
                str(cfg.target_endyear_ref))).mean('time')
        ds['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        ds['time'].data = ds['time'].dt.year.data
        ds_list.append(ds)
        i += 1
        # if i == 10:  # DEBUG
        #     break
    return xr.concat(ds_list, dim='model_ensemble')


def read_obs(cfg, change):
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
        regrid=cfg.target_diagnostic in REGRID_OBS)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        ds = ds.resample(time='1a').mean()
    ds = area_weighted_mean(ds)
    ds = ds.sel(time=slice('1950', None))
    if change:
        ds[varn] -= ds[varn].sel(time=slice(
            str(cfg.target_startyear_ref),
            str(cfg.target_endyear_ref))).mean('time')
    ds['time'].data = ds['time'].dt.year.data
    return ds


def read_weights(model_ensemble, cfg):
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds = xr.open_dataset(filename)
    ds = ds.sel(model_ensemble=model_ensemble)
    return ds['weights']


def plot(cfg, args):
    varn = cfg.target_diagnostic
    fig, ax = plt.subplots()#figsize=(15, 5))

    ds = read_data(cfg, change=args.change)
    data, xx = ds[varn].data, ds['time'].data

    l2 = None

    # --- unweighted baseline ---
    # for dd in data:
    #     ax.plot(xx, dd, color='gray', alpha=.2, lw=.5)  # all lines
    l1a = ax.fill_between(
        xx,
        quantile2(data, .05),
        quantile2(data, .95),
        # data.mean(axis=0) - data.std(axis=0),  # StdDev shading
        # data.mean(axis=0) + data.std(axis=0),
        color='gray',
        alpha=.2,
        zorder=100,
    )

    # plot mean
    [l1b] = ax.plot(
        xx, data.mean(axis=0),
        color='gray',
        lw=2,
        zorder=1000,
    )

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
    # def color(ww):
    #     """Different colors due to weights"""
    #     if ww > .05:
    #         return 'red'
    #     elif ww > .01:
    #         return 'orange'
    #     elif ww > .005:
    #         return 'yellow'
    #     return 'gray'
    for dd, ww in zip(data, weights):
        ww /= weights.max()
        ax.plot(
            xx, dd,
            color='darkred',  # color(ww),
            alpha=ww,
            lw=.3,
            zorder=10,
        )  # all lines
    [l3c] = ax.plot([], [], color='darkred', lw=.5)

    l3a = ax.fill_between(
        xx,
        quantile2(data, .05, weights),
        quantile2(data, .95, weights),
        # (np.average(data, weights=weights, axis=0) -
        #  np.sqrt(variance(data.swapaxes(0, 1), weights=weights))),
        # (np.average(data, weights=weights, axis=0) +
        #  np.sqrt(variance(data.swapaxes(0, 1), weights=weights))),
        color='darkred',
        alpha=.2,
        zorder=200,
    )
    [l3b] = ax.plot(
        xx, np.average(data, weights=weights, axis=0),
        color='darkred',
        lw=2,
        zorder=2000,
    )

    # --- observations ---
    obs = read_obs(cfg, change=args.change)
    if obs is not None:
        [l2] = ax.plot(
            obs['time'].data, obs[varn].data,
            color='k',
            lw=2,
            zorder=3000,
        )

    ax.set_xlabel('Year')
    ax.set_xlim(1950, 2060)

    # ax.set_ylabel('Temperature ($\degree$C)')
    if args.change:
        ax.set_ylim(-4, 8)
    else:
        ax.set_ylim(10, 30)

    ax.grid(zorder=0)
    if cfg.target_diagnostic == 'tas':
        varn = 'temperature ($\degree$C)'
    elif cfg.target_diagnostic == 'pr':
        varn = 'precipitation (mm/day)'
    else:
        varn = cfg.target_diagnostic

    region = np.atleast_1d(cfg.region)
    if 'NEU' in region and 'CEU' in region and 'MED' in region and len(region) == 3:
        region = 'EU'
    else:
        region = ', '.join(region)
    title = f'{region} {cfg.target_season} {varn}'
    if args.change:
        title += f' change ({cfg.target_startyear_ref}-{cfg.target_endyear_ref})'
    ax.set_title(title)

    if l2 is None:
        plt.legend([(l1a, l1b), (l3a, l3b)],
                   ['Unweighted mean & 90%', 'Weighted mean & 90%'],
                   loc='upper left')

    else:
        plt.legend(
            [(l1a, l1b),
             (l3a, l3b),
             l2,
             # l3c
            ],
            ['Unweighted mean & 90%',
             'Weighted mean & 90%',
             'Observations',
             # 'Models by weight'
            ],
            loc='upper left'
        )

    save_path = cfg.plot_path.replace('process_plots', 'timeseries')
    filename = os.path.join(save_path, 'lines_{}'.format(cfg.config))
    if args.change:
        filename += '_change'

    # filename = os.path.join(
    #     '/home/lukbrunn/Documents/Conferences_etal/20190204_EUCP-GA/figures',
    #     'tas_ceu_3')

    plt.savefig(filename + '.png', dpi=300)
    plt.savefig(filename + '.pdf', dpi=300)
    print('Saved: {}'.format(filename + '.png'))


def main():
    cfg, args = read_config()
    plot(cfg, args)


if __name__ == '__main__':
    main()
