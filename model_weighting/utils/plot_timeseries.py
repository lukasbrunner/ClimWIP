#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-10-11 09:17:55 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: A plot script for timeseries plots based on the output of the main model
 weighting script. Running this requires ClimWIP in the PYTHONPATH environment variable!
"""
import os
import argparse
import warnings
import numpy as np
import xarray as xr
import matplotlib as mpl
import regionmask
#mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from properscoring import crps_ensemble

from model_weighting.core.utils import read_config, log_parser
from model_weighting.core.utils_xarray import area_weighted_mean, quantile, flip_antimeridian

warnings.filterwarnings('ignore')

quantile = np.vectorize(quantile, signature='(n)->()', excluded=[1, 'weights', 'interpolation', 'old_style'])
period_ref = slice('1995', '2014')

PLOTPATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../plots/timeseries/')


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filename', type=str,
        help='Valid weights file (should end with .nc)')
    parser.add_argument(
        '--plot-type', '-t', dest='ext', default='png', type=str,
        help=' '.join([
            'A valid plot extension specifiying the file type. A special case',
            'is "show" which will call plt.show() instead of saving']))
    parser.add_argument(
        '--change-false', dest='change', action='store_false',
        help='Plot change instead of absolute values')
    parser.add_argument(
        '--perfect-model', '-p', dest='perfect_model_ensemble', type=str,
        default='ACCESS1-3_r1i1p1_CMIP5',
        help='String indicating the perfect model')
    args = parser.parse_args()

    return args


def read_models(ds, nan_mask, cfg, change):
    filenames = ds['filename'].data
    model_ensemble = ds['model_ensemble'].data

    ds_list = []
    for filename, model_ensemble in zip(filenames, model_ensemble):

        ds_var = xr.open_dataset(filename, use_cftime=True)
        scenario = filename.split('_')[-3]
        if 'ssp' in scenario:  # for cmip6 need to concat historical files
            filename = filename.replace(scenario, 'historical')
            ds_var2 = xr.open_dataset(filename, use_cftime=True)
            ds_var = xr.concat([ds_var2, ds_var], dim='time')

        ds_var = ds_var.drop(['height', 'file_qf'], errors='ignore')

        ds_var = ds_var.isel(time=ds_var['time.season'] == cfg.target_season)
        ds_var = ds_var.sel(time=slice('1950', '2100'))
        ds_var = ds_var.groupby('time.year').mean('time')

        ds_var = flip_antimeridian(ds_var)
        ds_var = ds_var.where(~nan_mask, drop=True)  # convert mask to index!

        ds_var = area_weighted_mean(ds_var)

        if change:
            ds_var -= ds_var.sel(year=slice(
                str(cfg.target_startyear_ref),
                str(cfg.target_endyear_ref))).mean('year')
        elif ds_var.max() > 100:  # assume Kelvin
            ds_var -= 273.15
            ds_var.attrs['unit'] = 'degC'

        ds_var['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        ds_list.append(ds_var)

    return xr.concat(ds_list, dim='model_ensemble')


def read_obs(ds, cfg, change):

    if cfg.obs_id is None:
        return None, None

    if isinstance(cfg.obs_id, str):
        cfg.obs_id = [cfg.obs_id]
        cfg.obs_path = [cfg.obs_path]

    ds_list = []
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):

        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(cfg.target_diagnostic, obs_id))
        ds_var = xr.open_dataset(filename, use_cftime=True)[cfg.target_diagnostic]

        if obs_id in ['CESM2', 'CESM2-2', 'CESM2-3']:
            filename = os.path.join(obs_path, '{}_mon_{}_g025_future.nc'.format(cfg.target_diagnostic, obs_id))
            ds_var2 = xr.open_dataset(filename, use_cftime=True)[cfg.target_diagnostic]

            ds_var = xr.concat([ds_var, ds_var2], dim='time')

        ds_var = ds_var.isel(time=ds_var['time.season'] == cfg.target_season)
        if len(obs_id) > 1:
            ds_var = ds_var.sel(time=period_ref)
        ds_var = ds_var.groupby('time.year').mean('time')

        ds_var = flip_antimeridian(ds_var)
        ds_var = ds_var.sel(lon=ds['lon'], lat=ds['lat'])

        if ds_var.max() > 100:  # assume Kelvin
            ds_var -= 273.15
            ds_var.attrs['unit'] = 'degC'

        ds_list.append(ds_var)

    ds_var = xr.concat(ds_list, dim='dataset')

    sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 0
    ds_var = ds_var.where(sea_mask)

    nan_mask = np.isnan(ds_var.mean('year', skipna=False).mean('dataset', skipna=False))
    ds_var = ds_var.where(~nan_mask)

    ds_var = area_weighted_mean(ds_var)

    if change:
        ds_var -= ds_var.sel(year=slice(
            cfg.target_startyear_ref,
            cfg.target_endyear_ref)).mean('year')

    return ds_var, nan_mask


def plot(ds_models, ds_obs, weights, args, ds):

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=.08, right=.97, top=.99)

    def plot_shading(yy1, yy2, xx=ds_models['year'].data, color='gray', alpha=.3, **kwargs):
        return ax.fill_between(
            xx, yy1, yy2,
            facecolor=color,
            edgecolor='none',
            alpha=alpha,
            zorder=100,
            **kwargs)

    def plot_line(yy, xx=ds_models['year'].data, color='gray', lw=2, **kwargs):
        return ax.plot(
            xx, yy,
            color=color,
            lw=lw,
            zorder=1000,
            **kwargs)[0]

    handles = []
    labels = []

    # --- baseline ---
    h1 = plot_shading(
        quantile(ds_models.data.swapaxes(0, 1), .25),
        quantile(ds_models.data.swapaxes(0, 1), .75))
    h2 = plot_line(np.mean(ds_models.data, axis=0))
    handles.append((h1, h2))
    labels.append('Mean & interquartile')

    # --- perfect model ---
    if 'perfect_model_ensemble' in weights.dims:
        model_ensemble = list(ds_models['model_ensemble'].data)
        model_ensemble.remove(args.perfect_model_ensemble)
        weights = weights.sel(model_ensemble=model_ensemble)

        plot_line(ds_models.sel(model_ensemble=args.perfect_model_ensemble))
        label = f'Perfect model: {args.perfect_model_ensemble}'

        # remove perfect model
        ds_models = ds_models.sel(model_ensemble=model_ensemble)

    # --- weighted ----
    assert np.all(weights['model_ensemble'] == ds_models['model_ensemble'])
    h1 = plot_shading(
        quantile(ds_models.data.swapaxes(0, 1), .25, weights=weights.data),
        quantile(ds_models.data.swapaxes(0, 1), .75, weights=weights.data),
        color='darkred')
    h2 = plot_line(
        np.average(ds_models.data, weights=weights.data, axis=0),
        color='darkred')
    handles.append((h1, h2))
    labels.append('Weighted mean & interquartile')

    # --- lines ---
    for dd, ww in zip(ds_models.data, weights.data):
        if ww >= sorted(weights)[-3]:
            color = 'darkred'
        elif ww < sorted(weights)[3]:
            color = 'darkviolet'
        else:
            continue
        plot_line(dd, color=color, lw=.2)

    h1 = ax.plot([], [], color='darkred', lw=1)[0]
    handles.append(h1)
    labels.append('Highest 3 models')

    h1 = ax.plot([], [], color='darkviolet', lw=1)[0]
    handles.append(h1)
    labels.append('Lowest 3 models')

    # --- observations ---
    if ds_obs is not None and len(ds_obs['dataset']) == 1:
        h1 = plot_line(ds_obs.data.squeeze(), xx=ds_obs['year'].data, color='k')
        label = 'Observations full range'
        # label = 'Pseudo observations: CESM2'  # TODO: !!
    elif ds_obs is not None:
        plot_shading(
            ds_obs.min('dataset', skipna=False),
            ds_obs.max('dataset', skipna=False),
            xx=ds_obs['year'].data,
            color='k', alpha=1)
        h1 = ax.plot([], [], color='k', lw=2)[0]  # use line for legend
        label = 'Observations full range'
    handles.append(h1)
    labels.append(label)

    # # try:
    # ax.axvspan(2081, 2100, alpha=.3, facecolor='darkblue', zorder=-999, edgecolor='none')

    # baseline = crps_ensemble(
    #     ds_obs.sel(year=slice('2081', '2100')).mean('year').data.squeeze(),
    #     ds_models.sel(year=slice('2081', '2100')).mean('year').data)
    # weighted = crps_ensemble(
    #     ds_obs.sel(year=slice('2081', '2100')).mean('year').data.squeeze(),
    #     ds_models.sel(year=slice('2081', '2100')).mean('year').data,
    #     weights=weights.data)
    # crps = (baseline - weighted) / baseline

    # ax.text(2091, 16.1, f'CRPS\n{crps:+.1%}', ha='center', va='bottom')
    # except Exception:
    #     pass

    # --- special and variable dependent settings ---
    # indicate and label different periods
    # colors = list(sns.color_palette('colorblind', 2))
    # ax.hlines([-1.5, -1.5, -1.5], [1995, 2041, 2081], [2014, 2060, 2100],
    #           ['k'] + colors, lw=2)
    # ax.text(2012, -1.4, 'reference', va='bottom', ha='right', fontsize='x-small')
    # ax.text(2050.5, -1.4, 'mid-century', va='bottom', ha='center', fontsize='x-small')
    # ax.text(2090.5, -1.4, 'end-of-century', va='bottom', ha='center', fontsize='x-small')

    # set y-axis limits and labels
    # ax.set_ylim(-2.5, 8.5)
    # title = f'{cfg.target_region} temperature anomaly ($\degree$C) relative to {cfg.target_startyear_ref}-{cfg.target_endyear_ref}'
    # ax.set_title(title)
    ax.set_ylabel('Temperature anomaly ($\degree$C)')

    ax.set_xlabel('Year')
    ax.set_xlim(1950, 2100)
    ax.grid(zorder=0, axis='y')

    plt.legend(handles, labels, loc='upper left')

    if args.ext == 'show':
        plt.show()
    else:
        os.makedirs(PLOTPATH, exist_ok=True)
        savename = os.path.join(os.path.join(PLOTPATH, ds.attrs['config']))
        if args.change:
            savename += '_change'
        if ds_obs is None:
            savename += f'_{args.perfect_model_ensemble}'
        plt.savefig(f'{savename}.{args.ext}', dpi=300)
        print(os.path.abspath(f'{savename}.{args.ext}'))


def main():
    args = read_input()
    ds = xr.open_dataset(args.filename)

    cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
    log_parser(cfg)

    ds_obs, nan_mask = read_obs(ds, cfg, args.change)
    ds_models = read_models(ds, nan_mask, cfg, args.change)

    plot(ds_models[cfg.target_diagnostic],
         ds_obs,
         ds['weights'],
         args,
         ds)


if __name__ == '__main__':
    main()
