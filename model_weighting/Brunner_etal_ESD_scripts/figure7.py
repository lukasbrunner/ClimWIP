#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2020 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted, ns

from model_weighting.core.get_filenames import get_filenames
from model_weighting.core.utils_xarray import area_weighted_mean, quantile, flip_antimeridian
from model_weighting.core.process_variants import get_model_variants
from model_weighting.core.utils import read_config, log_parser

quantile = np.vectorize(quantile, signature='(n)->()', excluded=[1, 'weights'])

LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FILENAME = 'tas_global_{ssp}_050_81-00.nc'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'


def extract_models_from_filenames(filenames):
    """
    Return the model name based on the given filenames.

    Parameters
    ----------
    filenames : str or list of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    models : str or ndarray of str based on input
    """
    if isinstance(filenames, str):
        return os.path.basename(filenames).split('_')[2]
    return np.array([os.path.basename(fn).split('_')[2] for fn in filenames])


def extract_variants_from_filenames(filenames):
    """
    Return the variant ID based on the given filenames.

    Parameters
    ----------
    filenames : str or list of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    variants : str or ndarray of str based on input
    """
    if isinstance(filenames, str):
        return os.path.basename(filenames).split('_')[4]
    return np.array([os.path.basename(fn).split('_')[4] for fn in filenames])


def intersect_models(*args):
    """
    Return intersection of filenames available for all cases.

    Parameters
    ----------
    *args : One or more list of filenames

    Returns
    -------
    *args : tuple of dict, same length as input
        One ore more dictionaries of filenames available for all models
        (and variants) from the input lists. The dictionary keys are
        <model>_<variant>
    """

    # if len(args) == 1:
    #     return args[0]

    filenames_dict = ()
    for filenames in args:
        models = extract_models_from_filenames(filenames)
        variants = extract_variants_from_filenames(filenames)
        filenames_dict += (
            {f'{model}_{variant}': filename
             for model, variant, filename in zip(models, variants, filenames)},)

    for filenames in filenames_dict:
        try:
            intersected_model_variants = list(
                np.intersect1d(intersected_model_variants, list(filenames.keys())))
        except NameError:
            intersected_model_variants = list(filenames.keys())

    args = ()
    for filenames in filenames_dict:
        args += ({key: filenames[key] for key in intersected_model_variants},)

    return args


def cluster_by_models(filenames, return_idx=False, return_keys=True, return_filenames=False):
    """
    Return a nested list of indices separating variants of the same model.

    Parameters
    ----------
    filenames : str or list of str or dict of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    nested : list of lists
    """
    if return_keys and not isinstance(filenames, dict):
        raise ValueError('filenames has to be dict if return_keys is True')

    if isinstance(filenames, str):
        filenames = [filenames]
    elif isinstance(filenames, dict):
        model_variants = np.array([*filenames.keys()])
        models = np.array([model_variant.split('_')[0]
                           for model_variant in model_variants])
        filenames = [filenames[key] for key in model_variants]
    else:
        models = extract_models_from_filenames(filenames)

    idx_nested = []
    for model in natsorted(np.unique(models), alg=ns.IC):
        idx = np.where(models == model)[0]
        idx_nested.append(idx)
    if return_idx:
        return idx_nested
    elif return_filenames:
        return [[filenames[idx] for idx in idxs] for idxs in idx_nested]
    return [[model_variants[idx] for idx in idxs] for idxs in idx_nested]


def create_pseudo_cfg(model_id, scenario, subset, path):
    cfg = type('', (), {})()
    cfg.performance_diagnostics = ['tas', 'psl']
    cfg.model_id = [model_id]
    cfg.model_scenario = [scenario]
    cfg.model_path = [path]
    cfg.variants_use = 'all'
    cfg.variants_select = 'natsorted'
    cfg.independence_diagnostics = None
    cfg.target_diagnostic = None
    cfg.subset = subset

    return cfg


def load_data(filenames, ssp):
    filenames, = intersect_models(filenames)
    model_ensemble_nested = cluster_by_models(filenames)

    model_list = []
    for model in model_ensemble_nested:
        variant_list = []
        for variant in model:
            filename = filenames[variant]
            ds = xr.open_dataset(filename)
            ds_hist = xr.open_dataset(filename.replace(ssp, 'historical'))
            ds = xr.concat([ds_hist, ds], dim='time')
            ds = ds['tas'].drop_vars('height', errors='ignore')
            ds = ds.sel(time=slice('1950', None))

            ds = ds.groupby('time.year').mean('time')
            ds = area_weighted_mean(ds)
            ds.data -= ds.sel(year=slice('1995', '2014')).mean('year').data
            variant_list.append(ds)
        ds = xr.concat(variant_list, dim='variant').mean('variant')
        model_id = '_'.join([
            variant.split('_')[0],
            str(len(model)) if len(model) > 1 else variant.split('_')[1],
            'CMIP6'])
        ds = ds.expand_dims({'model': [model_id]})
        model_list.append(ds)

    return xr.concat(model_list, dim='model')


def read_obs(ds, cfg, change):

    if cfg.obs_id is None:
        return None, None

    if not isinstance(cfg.obs_id, list):
        cfg.obs_id = [cfg.obs_id]
        cfg.obs_path = [cfg.obs_path]

    cfg.obs_id += ['BEST']
    cfg.obs_path += ['/net/h2o/climphys/lukbrunn/Data/InputData/BEST']

    ds_dict = {}
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(cfg.target_diagnostic, obs_id))
        ds_var = xr.open_dataset(filename, use_cftime=True).load()

        if 'TIME' in ds_var.dims:
            ds_var = ds_var.groupby('TIME').mean('TIME')
            ds_var = ds_var.rename({'TIME': 'year'})
        else:
            ds_var = ds_var.sel(time=slice('1975', None))
            ds_var = ds_var.groupby('time.year').mean('time')

        ds_var = ds_var[cfg.target_diagnostic]
        ds_var = area_weighted_mean(ds_var)

        if isinstance(ds_var['year'].data[0], float):
            ds_var.data = ds_var.data - ds_var.sel(year=slice(1995, 2014)).mean('year').data
        else:
            ds_var.data = ds_var.data - ds_var.sel(year=slice('1995', '2014')).mean('year').data
        ds_dict[obs_id] = ds_var
    return ds_dict


def load_weights(ssp):
    return xr.open_dataset(os.path.join(LOADPATH, FILENAME.format(
        ssp=ssp)))


def plot_shading(ax, xx, yy1, yy2, color='gray', edgecolor='none', alpha=.3, **kwargs):
    return ax.fill_between(
        xx, yy1, yy2,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        zorder=100,
        **kwargs)


def plot_line(ax, xx, yy, color='gray', lw=1, **kwargs):
    return ax.plot(
        xx, yy,
        color=color,
        lw=lw,
        zorder=1000,
        **kwargs)[0]


def statistics(ds, weights, ssp):
    ds_new = xr.Dataset(
        {'year': ('year', ds['year'].data, {'units': 'yyyy'}),
         'percentile': ('percentile',  [5, 16.67, 50, 83.33, 95, -99],
                        {'units': '%',
                         'description': 'Percentile values; last value (-99) represents the mean'})})

    data = np.empty((ds_new.dims['percentile'], ds_new.dims['year'])) * np.nan
    data_w = np.empty((ds_new.dims['percentile'], ds_new.dims['year'])) * np.nan
    for idx, qq in enumerate(ds_new['percentile'].data[:-1]):
        qq /= 100.
        data[idx] = quantile(ds.data.swapaxes(0, 1), qq)
        data_w[idx] = quantile(ds.data.swapaxes(0, 1), qq, weights=weights)
    data[-1] = np.ma.filled(np.ma.average(np.ma.masked_invalid(ds.data), axis=0))
    data_w[-1] = np.ma.filled(np.ma.average(np.ma.masked_invalid(ds.data), axis=0, weights=weights))

    ds_new['tas'] = xr.DataArray(data, dims=('percentile', 'year'),
                                 attrs={
                                     'standart_name': 'air_temperature',
                                     'long_name': 'Near-Surface Air Temperature',
                                     'units': 'K',
                                     'description': 'Mean Near-Surfave Air Temperature Anomaly relative to 1995-2014'})
    ds_new['tas_weighted'] = xr.DataArray(
        data_w, dims=('percentile', 'year'),
        attrs={
            'standart_name': 'air_temperature',
            'long_name': 'Near-Surface Air Temperature',
            'units': 'K',
            'description': 'Weighted Mean Near-Surfave Air Temperature Anomaly relative to 1995-2014'})

    ds_new['percentile'].attrs = {'units': '%',
                                  'description': 'Percentile values; last value (-99) represents the mean'}
    ds_new['year'].attrs = {'units': 'yyyy'}
    ds_new.attrs = {
        'description': 'Data prepared on request of Erich Fischer for use in the IPCC AR6. For more information see Brunner et al. 2020 ESDD',
        'reference': 'Brunner et al. 2020, DOI: https://doi.org/10.5194/esd-2020-23'}

    ds_new.to_netcdf(f'tas_{ssp}.nc')
    return ds_new

def plot(cmip6_ssp126, cmip6_ssp585, ds_obs,
         weights_ssp126=None, weights_ssp585=None):

    cmip6_ssp126 = statistics(cmip6_ssp126, weights_ssp126, ssp='ssp126')
    cmip6_ssp585 = statistics(cmip6_ssp585, weights_ssp585, ssp='ssp585')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.06, right=.97, top=.98, bottom=.05)

    # colors = sns.color_palette('Paired')
    colors = sns.color_palette('colorblind', 4)

    xx = cmip6_ssp126['year']

    handles = []
    labels = []

    # --- SSP126 ---
    if weights_ssp126 is not None:
        hh3 = plot_shading(
            ax, xx,
            cmip6_ssp126.isel(percentile=1)['tas'].data,
            cmip6_ssp126.isel(percentile=3)['tas'].data,
            # quantile(cmip6_ssp126.data.swapaxes(0, 1), 1/6.),
            # quantile(cmip6_ssp126.data.swapaxes(0, 1), 5/6.), sns.xkcd_rgb['greyish']
        )
        hh4 = plot_line(
            ax, xx,
            cmip6_ssp126.isel(percentile=-1)['tas'].data,
            # cmip6_ssp126.mean('model').data,
            # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5),
            sns.xkcd_rgb['greyish'])
        handles.append((hh3, hh4))
        labels.append('CMIP6 unweighted')

    hh1 = plot_shading(
        ax, xx.sel(year=slice('2014', None)),
        cmip6_ssp126.sel(year=slice('2014', None)).isel(percentile=1)['tas_weighted'].data,
        cmip6_ssp126.sel(year=slice('2014', None)).isel(percentile=3)['tas_weighted'].data,
        # quantile(cmip6_ssp126.sel(year=slice('2014', None)).data.swapaxes(0, 1), 1/6.,
        #          weights=weights_ssp126),
        # quantile(cmip6_ssp126.sel(year=slice('2014', None)).data.swapaxes(0, 1), 5/6.,
        #          weights=weights_ssp126)
        colors[0])
    hh2 = plot_line(
        ax, xx.sel(year=slice('2014', None)),
        cmip6_ssp126.sel(year=slice('2014', None)).isel(percentile=-1)['tas_weighted'].data,
        # np.average(cmip6_ssp126.sel(year=slice('2014', None)).data, axis=0, weights=weights_ssp126),
        # quantile(cmip6_ssp126.sel(year=slice('2014', None)).data.swapaxes(0, 1), .5,
                #  weights=weights_ssp126),
        colors[0])
    handles.append((hh1, hh2))
    labels.append('CMIP6 SSP1-2.6 weighted')

    # --- SSP585
    if weights_ssp585 is not None:
        plot_shading(
            ax, xx.sel(year=slice('2014', None)),
            cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=1)['tas'].data,
            cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=3)['tas'].data,
            # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), 1/6.),
            # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), 5/6.),
            sns.xkcd_rgb['greyish'])
        plot_line(
            ax, xx.sel(year=slice('2014', None)),
            cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=-1)['tas'].data,
            # cmip6_ssp585.sel(year=slice('2014', None)).mean('model').data,
            # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), .5),
            sns.xkcd_rgb['greyish'])

    hh1 = plot_shading(
        ax, xx.sel(year=slice('2014', None)),
        cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=1)['tas_weighted'].data,
        cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=3)['tas_weighted'].data,
        # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), 1/6.,
        #          weights=weights_ssp585),
        # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), 5/6.,
        #          weights=weights_ssp585),
        colors[3])
    hh2 = plot_line(
        ax, xx.sel(year=slice('2014', None)),
        cmip6_ssp585.sel(year=slice('2014', None)).isel(percentile=-1)['tas_weighted'].data,
        # np.average(cmip6_ssp585.sel(year=slice('2014', None)).data, axis=0, weights=weights_ssp585),
        # quantile(cmip6_ssp585.sel(year=slice('2014', None)).data.swapaxes(0, 1), .5, weights=weights_ssp585),
        colors[3])
    handles.append((hh1, hh2))
    labels.append('CMIP6 SSP5-8.5 weighted')

    ls = {
        'ERA5': '--',
        'MERRA2': ':',
        'BEST': '-',
    }
    # --- observations ---
    for key in ds_obs.keys():
        h1 = plot_line(ax, ds_obs[key]['year'].data, ds_obs[key].data, color='k', ls=ls[key])
        handles.append(h1)
        labels.append(key)

    # ax.set_xlabel('Year')
    ax.set_xlim(1975, 2100)
    ax.set_ylim(-1, 6)
    ax.grid(zorder=0, axis='y')

    ax.hlines([1., 0., 0.], [1980, 2041, 2081], [2014, 2060, 2100], color='k', lw=2)
    ax.text(1996.5, 1.1, 'Diagnostic period', va='bottom', ha='center')
    ax.text(2050.5, .1, 'Mid-century', va='bottom', ha='center')
    ax.text(2090.5, .1, 'End-of-century', va='bottom', ha='center')

    ax.set_ylabel(u'Temperature change (°C) relative to 1995-2014')

    leg = plt.legend(handles, labels, loc='upper left', ncol=2, title='Mean and likely (66\%) range')
    leg._legend_box.align = 'left'

    plt.savefig(f'figures/figure7.png', dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure7.pdf'), dpi=300)


def main():
    weights_ssp126 = load_weights('ssp126')
    weights_ssp585 = load_weights('ssp585')

    cmip6_ssp126 = load_data(weights_ssp126['filename'].data, 'ssp126')
    cmip6_ssp585 = load_data(weights_ssp585['filename'].data, 'ssp585')

    models = weights_ssp126['model'].data

    weights_ssp126 = weights_ssp126.sel(model=models)
    weights_ssp585 = weights_ssp585.sel(model=models)

    cmip6_ssp126 = cmip6_ssp126.sel(model=models)
    cmip6_ssp585 = cmip6_ssp585.sel(model=models)

    cfg = read_config(weights_ssp126.attrs['config'], weights_ssp126.attrs['config_path'])
    ds_obs = read_obs(weights_ssp126, cfg, True)

    plot(cmip6_ssp126,
         cmip6_ssp585,
         ds_obs,
         weights_ssp126=weights_ssp126['weights_mean'],
         weights_ssp585=weights_ssp585['weights_mean'],
    )


if __name__ == '__main__':
    main()
