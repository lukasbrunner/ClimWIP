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
from matplotlib.gridspec import GridSpec
import seaborn as sns
from natsort import natsorted, ns

from model_weighting.core.get_filenames import get_filenames
from model_weighting.core.utils_xarray import area_weighted_mean, quantile
from model_weighting.core.process_variants import get_model_variants

quantile = np.vectorize(quantile, signature='(n)->()', excluded=[1, 'weights'])

SAVEPATH = '/home/lukbrunn/Documents/Scripts/climWIP_clean_paper/model_weighting/scripts_paper/revision1/figures'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FILENAME = 'tas_global_pseudo_obs_{ssp}_050_MIROC-ESM-r1i1p1-CMIP5-{rcp}_81-00_ind.nc'


def create_pseudo_cfg(model_id, scenario, subset, path):
    cfg = type('', (), {})()
    cfg.performance_diagnostics = ['tas']
    cfg.model_id = [model_id]
    cfg.model_scenario = [scenario]
    cfg.model_path = [path]
    cfg.variants_use = 'all'
    cfg.variants_select = 'natsorted'
    cfg.independence_diagnostics = None
    cfg.target_diagnostic = None
    cfg.subset = subset

    return cfg

def remove_models(dict_, key_patterns_remove=[]):
    if isinstance(key_patterns_remove, str):
        key_patterns_remove = [key_patterns_remove]
    dict_new = {}
    for key in dict_.keys():
        if np.any([key_pattern in key for key_pattern in key_patterns_remove]):
            continue
        dict_new[key] = dict_[key]

    return dict_new

def load_data(*args, filenames=None):
    cfg = create_pseudo_cfg(*args)
    if filenames is None:
        filenames = get_filenames(cfg)['tas']
    else:
        filenames = {key: str(filenames.sel(model_ensemble=key).data) for key in filenames['model_ensemble'].data}

    if len(filenames) > 1:
        filenames = remove_models(filenames, 'MIROC')

    model_ensemble_nested = get_model_variants([*filenames.keys()])

    model_list = []
    for model in model_ensemble_nested:
        variant_list = []
        for variant in model:
            filename = filenames[variant]
            ds = xr.open_dataset(filename)
            if args[0] == 'CMIP6':
                ds_hist = xr.open_dataset(filename.replace(args[1], 'historical'))
                ds = xr.concat([ds_hist, ds], dim='time')
            ds = ds['tas'].drop_vars('height', errors='ignore')
            ds = ds.sel(time=slice('1950', '2100'))
            ds = ds.groupby('time.year').mean('time')
            ds = area_weighted_mean(ds)
            ds.data -= ds.sel(year=slice('1995', '2014')).mean('year').data
            variant_list.append(ds)
        ds = xr.concat(variant_list, dim='variant').mean('variant')
        if '_CMIP6' in model[0]:
            model_id = '_'.join([
                model[0].split('_')[0],
                str(len(model)) if len(model) > 1 else model[0].split('_')[1],
                model[0].split('_')[2]])
        else:
            model_id = model[0].split('_')[0]
        ds = ds.expand_dims({'model': [model_id]})
        model_list.append(ds)

    return xr.concat(model_list, dim='model')


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


def plot_single(ax, cmip5_rcp26, cmip6_ssp126, cmip5_rcp85, cmip6_ssp585,
                weights_ssp126=None, weights_ssp585=None, xticks=False,
                weights_ssp126_tasTREND=None, weights_ssp585_tasTREND=None):

    colors = sns.color_palette('colorblind', 4)
    xx = cmip5_rcp26['year']
    xx2 = cmip5_rcp26['year'].sel(year=slice('2015', None))

    handles = []
    labels = []
    ### --- RCP26&SSP126 ---
    # --- pseudo obs ---
    hh = plot_line(
        ax, xx,
        cmip5_rcp26.data.squeeze(),
        colors[0], ls='--')
    handles.append(hh)
    labels.append('Pseudo observation: historical \& RCP2.6')

    # --- unweighted ---
    hh1 = plot_shading(
        ax, xx,
        quantile(cmip6_ssp126.data.swapaxes(0, 1), 1/6.),
        quantile(cmip6_ssp126.data.swapaxes(0, 1), 5/6.), sns.xkcd_rgb['greyish'])
    hh2 = plot_line(
        ax, xx,
        cmip6_ssp126.mean('model').data,
        # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5),
        sns.xkcd_rgb['greyish'])

    handles.append((hh1, hh2))
    labels.append('CMIP6 mean \& 66\,\% range unweighted')

    # --- weighted ---
    hh1 = plot_shading(
        ax, xx2,
        quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
                 weights=weights_ssp126),
        quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
                 weights=weights_ssp126), colors[0])
    hh2 = plot_line(
        ax, xx2,
        np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp126.sel(year=slice('2015', None)).data),
                                   axis=0, weights=weights_ssp126)),
        # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5, weights=weights_ssp126),
        colors[0])

    handles.append((hh1, hh2))
    labels.append('CMIP6 SSP1-2.6 mean \& 66\,\% range weighted')

    if weights_ssp126_tasTREND is not None:
        # --- weighted ---
        # hh1 = plot_shading(
        #     ax, xx2,
        #     quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
        #              weights=weights_ssp126_tasTREND),
        #     quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
        #              weights=weights_ssp126_tasTREND), colors[0], alpha=.15)
        hh2 = plot_line(
            ax, xx2,
            np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp126.sel(year=slice('2015', None)).data),
                                       axis=0, weights=weights_ssp126_tasTREND)),
            # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5, weights=weights_ssp126_tasTREND),
            colors[0], ls=':')

        handles.append((hh2))
        labels.append('CMIP6 SSP1-2.6 mean weighted (tasTREND only)')

    ### --- RCP85/SSP585 ---
    # --- pseudo obs ---
    hh = plot_line(
        ax, xx,
        cmip5_rcp85.data.squeeze(),
        colors[3], ls='--')

    handles.append(hh)
    labels.append('Pseudo observation: historical \& RCP8.5')

    # --- unweighted ---
    plot_shading(
        ax, xx,
        quantile(cmip6_ssp585.data.swapaxes(0, 1), 1/6.),
        quantile(cmip6_ssp585.data.swapaxes(0, 1), 5/6.), sns.xkcd_rgb['greyish'])
    plot_line(
        ax, xx,
        cmip6_ssp585.mean('model').data,
        # quantile(cmip6_ssp585.data.swapaxes(0, 1), .5),
        sns.xkcd_rgb['greyish'])

    # --- weighted ---
    hh1 = plot_shading(
        ax, xx2,
        quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
                 weights=weights_ssp585),
        quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
                 weights=weights_ssp585), colors[3])
    hh2 = plot_line(
        ax, xx2,
        np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp585.sel(year=slice('2015', None)).data),
                                   axis=0, weights=weights_ssp585)),
        # quantile(cmip6_ssp585.data.swapaxes(0, 1), .5, weights=weights_ssp585),
        colors[3])

    handles.append((hh1, hh2))
    labels.append('CMIP6 SSP5-8.5 mean \& 66\,\% range weighted')

    if weights_ssp585_tasTREND is not None:
        # --- weighted ---
        # hh1 = plot_shading(
        #     ax, xx2,
        #     quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
        #              weights=weights_ssp585_tasTREND),
        #     quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
        #              weights=weights_ssp585_tasTREND), colors[3], alpha=.15)
        hh2 = plot_line(
            ax, xx2,
            np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp585.sel(year=slice('2015', None)).data),
                       axis=0, weights=weights_ssp585_tasTREND)),
            # quantile(cmip6_ssp585.data.swapaxes(0, 1), .5, weights=weights_ssp585_tasTREND),
            colors[3], ls=':')

        handles.append((hh2))
        labels.append('CMIP6 SSP5-8.5 mean weighted (tasTREND only)')

    ax.set_xlim(1975, 2100)
    ax.set_ylim(-1, 6)
    ax.grid(zorder=0, axis='y')

    ax.set_yticks([0, 2, 4])
    ax.set_xticks([1980, 2020, 2060, 2100])

    handles = np.array(handles)[np.array([0, 3, 1, 2, 4])]
    labels = np.array(labels)[np.array([0, 3, 1, 2, 4])]
    # handles = np.array(handles)[np.array([0, 4, 1, 2, 3, 5, 6])]
    # labels = np.array(labels)[np.array([0, 4, 1, 2, 3, 5, 6])]
    leg = ax.legend(handles, labels, loc='upper left')  # , title='Median, 50%, 90%')
    # leg._legend_box.align = 'left'


def main():

    weights_ssp126 = xr.open_dataset(
        os.path.join(LOADPATH, FILENAME.format(ssp='ssp126', rcp='RCP26')))
    cmip5_rcp26 = load_data('CMIP5', 'rcp26', ['MIROC-ESM_r1i1p1_CMIP5'], '/net/atmos/data/cmip5-ng')
    cmip6_ssp126 = load_data('CMIP6', 'ssp126', None, '/net/ch4/data/cmip6-Next_Generation',
                             filenames=weights_ssp126['filename'])
    weights_ssp126 = weights_ssp126['weights_mean']
    models = np.intersect1d(weights_ssp126['model'].data, cmip6_ssp126['model'])
    weights_ssp126 = weights_ssp126.sel(model=models)
    cmip6_ssp126 = cmip6_ssp126.sel(model=models)

    assert len(models) == 31

    weights_ssp585 = xr.open_dataset(
        os.path.join(LOADPATH, FILENAME.format(ssp='ssp585', rcp='RCP85')))
    cmip5_rcp85 = load_data('CMIP5', 'rcp85', ['MIROC-ESM_r1i1p1_CMIP5'], '/net/atmos/data/cmip5-ng')
    cmip6_ssp585 = load_data('CMIP6', 'ssp585', None, '/net/ch4/data/cmip6-Next_Generation',
                             filenames=weights_ssp585['filename'])
    weights_ssp585 = weights_ssp585['weights_mean']
    models = np.intersect1d(weights_ssp585['model'].data, cmip6_ssp585['model'].data)
    weights_ssp585 = weights_ssp585.sel(model=models)
    cmip6_ssp585 = cmip6_ssp585.sel(model=models)

    assert len(models) == 31

    weights_ssp126_tasTREND = xr.open_dataset(
        os.path.join(
            LOADPATH,
            'tas_global_pseudo_obs_{ssp}_tasTREND_MIROC-ESM-r1i1p1-CMIP5-{rcp}_81-00_ind.nc'.format(
                ssp='ssp126', rcp='RCP26')))['weights_mean']
    weights_ssp126_tasTREND = weights_ssp126_tasTREND.sel(model=models)

    weights_ssp585_tasTREND = xr.open_dataset(
        os.path.join(
            LOADPATH,
            'tas_global_pseudo_obs_{ssp}_tasTREND_MIROC-ESM-r1i1p1-CMIP5-{rcp}_81-00_ind.nc'.format(
                ssp='ssp585', rcp='RCP85')))['weights_mean']
    weights_ssp585_tasTREND = weights_ssp585_tasTREND.sel(model=models)

    fig, ax = plt.subplots(figsize=(9*.83, 6*.83))
    fig.subplots_adjust(left=.07, right=.97, bottom=.05, top=.94)

    plot_single(
        ax,
        cmip5_rcp26, cmip6_ssp126,
        cmip5_rcp85, cmip6_ssp585,
        weights_ssp126=weights_ssp126,
        weights_ssp585=weights_ssp585,
        # weights_ssp126_tasTREND=weights_ssp126_tasTREND,
        # weights_ssp585_tasTREND=weights_ssp585_tasTREND,
        xticks=False)

    ax.hlines([1., 0., 0.], [1980, 2041, 2081], [2014, 2060, 2100], color='k', lw=2)
    ax.text(1996.5, 1.1, 'Diagnostic period', va='bottom', ha='center')
    ax.text(2050.5, .1, 'Target period', va='bottom', ha='center')
    ax.text(2090.5, .1, 'Target period', va='bottom', ha='center')

    ax.set_ylabel('Temperature change (°C) relative to 1995-2014')

    ax.set_title('\\textbf{(a) Combined weighting based on pseudo-observations from MIROC-ESM (CMIP5)}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join(SAVEPATH, 'figure2a.png'), dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure2a.pdf'), dpi=300)


if __name__ == '__main__':
    main()
