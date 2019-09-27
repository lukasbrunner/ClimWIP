#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-09-27 17:11:34 lukbrunn>

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
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model_weighting.core.utils import read_config, log_parser
from model_weighting.core.utils_xarray import area_weighted_mean, quantile
from model_weighting.core.diagnostics import calculate_basic_diagnostic

quantile = np.vectorize(quantile, signature='(n)->()', excluded=[1, 'weights', 'interpolation', 'old_style'])
period_ref = slice('1995', '2014')

# TODO: add this to parser
perfect_model_ensemble = 'ACCESS1-3_r1i1p1_CMIP5'

PLOTPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/timeseries/'


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='config', nargs='?', default='med_tas_81-00',
        help='Name of the configuration to use (optional; default: med_tas_81-00).')
    parser.add_argument(
        '--filename', '-f', dest='filename', default='../configs/paper/tas.ini',
        help='Relative or absolute path/filename.ini of the config file.')
    parser.add_argument(
        '--plot-type', '-t', dest='ext', default='png', type=str,
        help=' '.join([
            'A valid plot extension specifiying the file type. A special case',
            'is "show" which will call plt.show() instead of saving']))
    parser.add_argument(
        '--change-false', dest='change', action='store_false',
        help='Plot change instead of absolute values')
    args = parser.parse_args()
    cfg = read_config(args.config, args.filename)
    log_parser(cfg)
    return cfg, args


def read_models(filenames, mask, change, cfg):
    varn = cfg.target_diagnostic
    ds_list = []
    i = 0
    for filename, model_ensemble in zip(
            filenames.data, filenames.model_ensemble.data):

        ds = calculate_basic_diagnostic(
            filename,
            varn,
            outfile=None,
            id_=cfg.model_id,
            time_period=None,
            season=cfg.target_season,
            time_aggregation=None,
            mask_ocean=cfg.target_masko,
            region=cfg.target_region)

        if mask is not None:
            ds[cfg.target_diagnostic] = xr.apply_ufunc(
                apply_obs_mask, ds[cfg.target_diagnostic], mask,
                input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                output_core_dims=[['lat', 'lon']],
                vectorize=True)

        # xarray spams warnings about the masked vales
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            ds = ds.resample(time='1A').mean()
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


def apply_obs_mask(data, mask):
    data = np.ma.masked_array(data, mask)  # mask data
    data = np.ma.filled(data, fill_value=np.nan)  # set masked to NaN
    return data


def read_obs(cfg, change):

    if cfg.obs_id is None:
        return None, None

    if isinstance(cfg.obs_id, str):
        cfg.obs_id = [cfg.obs_id]
        cfg.obs_path = [cfg.obs_path]

    varn = cfg.target_diagnostic
    ds_list = []
    mask = False
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(varn, obs_id))

        ds = calculate_basic_diagnostic(
            filename,
            varn,
            outfile=None,
            id_=obs_id,
            time_period=None,
            season=cfg.target_season,
            time_aggregation=None,
            mask_ocean=cfg.target_masko,
            region=cfg.target_region)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            ds = ds.resample(time='1A').mean()

        mask = np.ma.mask_or(mask, np.isnan(ds.mean('time', skipna=False)[cfg.target_diagnostic]))
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
        output_core_dims=[['lat', 'lon']],
        vectorize=True)

    # ds[cfg.target_diagnostic].data = ds[cfg.target_diagnostic].data > 0.05
    return area_weighted_mean(ds), mask


def read_weights(model_ensemble, cfg):
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds = xr.open_dataset(filename)
    ds = ds.sel(model_ensemble=model_ensemble)
    return ds['weights']


def plot(ds_models, ds_obs, weights, args, cfg):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=.08, right=.97, top=.99)

    handles = []
    labels = []

    # --- unweighted baseline ---
    l1a = ax.fill_between(
        ds_models['time'].data,
        quantile(ds_models.data.swapaxes(0, 1), .25),
        quantile(ds_models.data.swapaxes(0, 1), .75),
        facecolor='gray',
        edgecolor='none',
        alpha=.5,
        zorder=100,
    )
    # plot mean
    [l1b] = ax.plot(
        ds_models['time'].data,
        np.mean(ds_models.data, axis=0),
        color='gray',
        lw=2,
        zorder=1000,
    )
    handles.append((l1a, l1b))
    labels.append('Mean & interquartile')

    # --- weighted ----

    # --- perfect model ---
    assert np.all(weights['model_ensemble'] == ds_models['model_ensemble'])
    if 'perfect_model_ensemble' in weights.dims:
        weights = weights.sel(perfect_model_ensemble=perfect_model_ensemble)
        model_ensemble = list(ds_models['model_ensemble'].data)
        model_ensemble.remove(perfect_model_ensemble)
        weights = weights.sel(model_ensemble=model_ensemble)

        # plot 'true' model
        [l2] = ax.plot(ds_models['time'].data,
                       ds_models.sel(model_ensemble=perfect_model_ensemble).data,
                       color='k', lw=2, zorder=3000)

        label = f'Perfect model: {perfect_model_ensemble}'

        # remove perfect model
        ds_models = ds_models.sel(model_ensemble=model_ensemble)

    l3a = ax.fill_between(
        ds_models['time'].data,
        quantile(ds_models.data.swapaxes(0, 1), .25, weights=weights.data),
        quantile(ds_models.data.swapaxes(0, 1), .75, weights=weights.data),
        facecolor='darkred',
        edgecolor='none',
        alpha=.2,
        zorder=200,
    )
    [l3b] = ax.plot(
        ds_models['time'].data,
        np.average(ds_models.data, weights=weights.data, axis=0),
        color='darkred',
        lw=2,
        zorder=2000,
    )

    handles.append((l3a, l3b))
    labels.append('Weighted mean & interquartile')

    def color(ww, ww_all):
        """Different colors due to weights"""

        # if ww > np.percentile(ww_all, 90):
        #     return 'darkred'
        # elif ww < np.percentile(ww_all, 10):
        #     return 'darkviolet'
        # return 'none'

        if ww >= sorted(ww_all)[-3]:
            return 'darkred'
        elif ww < sorted(ww_all)[3]:
            return 'darkviolet'
        return 'none'

    for dd, ww in zip(ds_models.data, weights.data):
        ax.plot(
            ds_models['time'].data,
            dd,
            color=color(ww, weights),
            lw=.2,
            zorder=10,
        )  # all lines

    [ll] = ax.plot([], [], color='darkred', lw=1)
    handles.append(ll)
    labels.append('Highest 3 models')

    [ll] = ax.plot([], [], color='darkviolet', lw=1)
    handles.append(ll)
    labels.append('Lowest 3 models')

    # --- observations ---
    if ds_obs is not None and len(ds_obs['dataset']) == 1:
        if 'dataset_dim' in ds_obs.dims:
            oo = ds_obs.isel(dataset_dim=0)
        elif 'dataset' in ds_obs.dims:
            oo = ds_obs.isel(dataset=0)
        else:
            oo = ds_obs

        try:
            [l2] = ax.plot(
                ds_obs['time'].data,
                oo.data,
                color='k',
                lw=2,
                zorder=3000,
        )
        except Exception:
            import ipdb; ipdb.set_trace()
        label = 'Observations full range'
    elif ds_obs is not None:
        min_ = ds_obs.min('dataset', skipna=False)
        max_ = ds_obs.max('dataset', skipna=False)
        l2 = ax.fill_between(
            ds_obs['time'].data,
            min_.data,
            max_.data,
            color='k',
            zorder=10,
        )
        [l2] = ax.plot([], [], color='k', lw=2)  # use line for legend
        label = 'Observations full range'
    handles.append(l2)
    labels.append(label)

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
        savename = os.path.join(os.path.join(PLOTPATH, args.config))
        if args.change:
            savename += '_change'
        if ds_obs is None:
            savename += f'_{perfect_model_ensemble}'
        plt.savefig(f'{savename}.{args.ext}', dpi=300)
        print(os.path.abspath(f'{savename}.{args.ext}'))


def main():
    cfg, args = read_input()

    # read results file
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds_results = xr.open_dataset(filename)

    ds_obs, mask = read_obs(cfg, args.change)
    ds_models = read_models(ds_results['filename'], mask, args.change, cfg)

    plot(ds_models[cfg.target_diagnostic],
         ds_obs[cfg.target_diagnostic] if ds_obs is not None else None,
         ds_results['weights'],
         args,
         cfg)


if __name__ == '__main__':
    main()
