#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-03-15 10:55:59 lukbrunn>

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
from utils_python.decorators import vectorize
from utils_python.xarray import area_weighted_mean
from utils_python.math import variance, quantile2, std

from model_weighting.model_weighting import set_up_filenames, get_filenames
from model_weighting.functions.diagnostics import calculate_basic_diagnostic

REGION_DIR = '{}/../cdo_data/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'

variance = np.vectorize(variance, signature='(n)->()', excluded=['weights', 'biased'])
period_ref = slice('1995', '2014')
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


def read_data(cfg, mask, change):
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
            region=cfg.target_region)

        ds[cfg.target_diagnostic] = xr.apply_ufunc(
            apply_obs_mask, ds[cfg.target_diagnostic], mask,
            input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
            output_core_dims=[['lat', 'lon']])

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
    return xr.concat(ds_list, dim='model_ensemble')


@vectorize('(n,m),(n,m)->(n,m)')
def apply_obs_mask(data, mask):
    data = np.ma.masked_array(data, mask)  # mask data
    data = np.ma.filled(data, fill_value=np.nan)  # set masked to NaN
    return data


def read_obs(cfg, change):

    if cfg.obsdata is None:
        return None

    if isinstance(cfg.obsdata, str):
        cfg.obsdata = [cfg.obsdata]
        cfg.obs_path = [cfg.obs_path]

    varn = cfg.target_diagnostic
    ds_list = []
    mask = False
    for obs_path, obsdata in zip(cfg.obs_path, cfg.obsdata):
        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(varn, obsdata))

        ds = calculate_basic_diagnostic(
            filename, varn,
            outfile=None,
            time_period=None,
            season=cfg.target_season,
            time_aggregation=None,
            mask_ocean=cfg.target_masko,
            region=cfg.target_region,
            regrid=cfg.target_diagnostic in REGRID_OBS)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            ds = ds.resample(time='1a').mean()

        mask = np.ma.mask_or(mask, np.isnan(ds.mean('time', skipna=False)[cfg.target_diagnostic]))

        ds = ds.sel(time=slice('1995', '2014'))
        if change:
            ds[varn] -= ds[varn].sel(time=slice(
                str(cfg.target_startyear_ref),
                str(cfg.target_endyear_ref))).mean('time')
        ds['time'].data = ds['time'].dt.year.data
        ds_list.append(ds)
    ds = xr.concat(ds_list, dim='dataset')

    # before taking the area mean set all grid points to NaN which are NaN in
    # ANY of the observational datasets
    ds[cfg.target_diagnostic] = xr.apply_ufunc(
        apply_obs_mask, ds[cfg.target_diagnostic], mask,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
        output_core_dims=[['lat', 'lon']])
    return area_weighted_mean(ds), mask


def read_weights(model_ensemble, cfg):
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds = xr.open_dataset(filename)
    ds = ds.sel(model_ensemble=model_ensemble)
    return ds['weights']


def plot(cfg, args):
    varn = cfg.target_diagnostic
    fig, ax = plt.subplots()  # figsize=(15, 5))

    obs, mask = read_obs(cfg, change=args.change)
    ds = read_data(cfg, mask, change=args.change)
    data, xx = ds[varn].data, ds['time'].data

    handles = []
    labels = []

    # --- unweighted baseline ---
    # for dd in data:
    #     ax.plot(xx, dd, color='gray', alpha=.2, lw=.5)  # all lines
    l1a = ax.fill_between(
        xx,
        quantile2(data, .05),
        quantile2(data, .95),
        color='gray',
        alpha=.2,
        zorder=100,
    )

    # l1a = ax.fill_between(
    #     xx,
    #     data.mean(axis=0) - .5*data.std(axis=0),  # StdDev shading
    #     data.mean(axis=0) + .5*data.std(axis=0),
    #     color='gray',
    #     alpha=.2,
    #     zorder=100,
    # )

    # plot mean
    [l1b] = ax.plot(
        xx, data.mean(axis=0),
        color='gray',
        lw=2,
        zorder=1000,
    )

    handles.append((l1a, l1b))
    labels.append('Mean & 90% spread')

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

        handles.append(l2)
        labels.append('Perfect model')

    weights = weights.data

    l3a = ax.fill_between(
        xx,
        quantile2(data, .05, weights),
        quantile2(data, .95, weights),
        color='darkred',
        alpha=.2,
        zorder=200,
    )

    # l3a = ax.fill_between(
    #     xx,
    #     np.average(data, weights=weights, axis=0) - .5*std(data, weights),
    #     np.average(data, weights=weights, axis=0) + .5*std(data, weights),
    #     color='darkred',
    #     alpha=.2,
    #     zorder=200,
    # )

    [l3b] = ax.plot(
        xx, np.average(data, weights=weights, axis=0),
        color='darkred',
        lw=2,
        zorder=2000,
    )

    handles.append((l3a, l3b))
    labels.append('Weighted mean & 90% spread')

    # def color(ww):
    #     """Different colors due to weights"""
    #     if ww > 1.5:
    #         return 'darkred'
    #     elif ww > 1.:
    #         return 'darkorange'
    #     elif ww > .5:
    #         return 'none'
    #     return 'none'

    def color(ww, ww_all):
        """Different colors due to weights"""

        if ww > np.percentile(ww_all, 90):
            return 'darkred'
        # elif ww > np.percentile(ww_all, 75):
        #     return 'darkorange'
        elif ww < np.percentile(ww_all, 10):
            return 'darkviolet'
        return 'none'

    for dd, ww in zip(data, weights):
        # ww /= weights.max()
        ax.plot(
            xx, dd,
            color=color(ww, weights),
            # color='darkred',  # color(ww),
            # alpha=ww,
            lw=.2,
            zorder=10,
        )  # all lines

    [ll] = ax.plot([], [], color='darkred', lw=1)
    handles.append(ll)
    labels.append('Best 10% of models')

    # [ll] = ax.plot([], [], color='darkorange', lw=1)
    # handles.append(ll)
    # labels.append('Best 25% of models')

    [ll] = ax.plot([], [], color='darkviolet', lw=1)
    handles.append(ll)
    labels.append('Worst 10% of models')

    # --- observations ---
    if obs is not None and len(obs['dataset']) == 1:
        if 'dataset_dim' in obs.dims:
            oo = obs.isel(dataset_dim=0)
        elif 'dataset' in obs.dims:
            oo = obs.isel(dataset=0)
        else:
            oo = obs

        try:
            [l2] = ax.plot(
                obs['time'].data,
                oo[varn].data,
                color='k',
                lw=2,
                zorder=3000,
        )
        except Exception:
            import ipdb; ipdb.set_trace()
    elif obs is not None:
        min_ = obs.min('dataset', skipna=False)
        max_ = obs.max('dataset', skipna=False)
        l2 = ax.fill_between(
            obs['time'].data,
            min_[varn].data,
            max_[varn].data,
            color='k',
            zorder=3000,
        )
        [l2] = ax.plot([], [], color='k', lw=2)  # use line for legend

    handles.append(l2)
    labels.append('Observations full range')

    ax.set_xlabel('Year')
    ax.set_xlim(1950, 2100)

    # if args.change and cfg.target_diagnostic == 'tas':
    #     ax.set_ylim(-4, 8)
    # elif not args.change and cfg.target_diagnostic == 'tas':
    #     ax.set_ylim(10, 30)

    ax.grid(zorder=0)
    if cfg.target_diagnostic == 'tas':
        varn = 'temperature ($\degree$C)'
    elif cfg.target_diagnostic == 'pr':
        varn = 'precipitation (mm/day)'
    else:
        varn = cfg.target_diagnostic

    title = f'{cfg.target_region} {cfg.target_season} {varn}'
    if args.change:
        title += f' change ({cfg.target_startyear_ref}-{cfg.target_endyear_ref})'
    ax.set_title(title)

    plt.legend(handles, labels, loc='upper left')

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
