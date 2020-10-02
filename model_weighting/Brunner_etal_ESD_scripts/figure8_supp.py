#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import numpy as np
import xarray as xr
import regionmask
import matplotlib as mpl
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from model_weighting.core.utils_xarray import area_weighted_mean, quantile
from boxplot import boxplot

LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FN_PATTERN = 'tas_global_{ssp}_{trend}_{period}_bootstrap-*.nc'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'

def load_bootstrap(ssp='ssp126', trend='050', period='41-60'):

    filenames = glob(os.path.join(LOADPATH, FN_PATTERN.format(ssp=ssp, trend=trend, period=period)))
    ds_list = []
    for fn in filenames:
        ds = xr.open_dataset(fn)
        ds = ds.assign(model_ensemble = lambda xx: [x.split('_')[0] for x in xx.model_ensemble.data])
        ds = ds.assign_coords({'model_ensemble': np.array([mm.split('_')[0] for mm in ds['model_ensemble'].data])})
        ds = ds.rename({'model_ensemble': 'model'})
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='sample')
    return ds


def get_factor(arr):
    arr = arr * len(arr)
    arr[arr < 1.] = 1 - 1/arr[arr < 1.]
    arr[arr >= 1.] -= 1
    return arr


def sort_by_model(ds1, ds2):
    models1 = ds1['model'].data
    models2 = ds2['model'].data

    models = np.intersect1d(models1, models2)

    ds1 = ds1.sel(model=models)
    ds2 = ds2.sel(model=models)

    return ds1, ds2


def calculate_quantiles(data, weights=None):
    qq = quantile(data, (.05, 1/6, .5, 5/6., .95), weights=weights)
    qq = np.concatenate([qq, [np.average(data, weights=weights)]])
    return qq


def plot_distributions(ax, xx, ds, ds_bt, color, color1):

    ds_bt = area_weighted_mean(ds_bt)
    ds = area_weighted_mean(ds)

    dist = xr.apply_ufunc(
        calculate_quantiles,
        ds_bt['tas'], [None],
        input_core_dims=[['model'], []],
        output_core_dims=[['quantile']],
        vectorize=True)

    distw = xr.apply_ufunc(
        calculate_quantiles,
        ds_bt['tas'], ds_bt['weights'],
        input_core_dims=[['model'], ['model']],
        output_core_dims=[['quantile']],
        vectorize=True)

    boxplot(
        ax, xx,
        mean=dist.median('sample')[-1],
        box=(dist.median('sample')[1], dist.median('sample')[3]),
        whis=(dist.median('sample')[0], dist.median('sample')[4]),
        color=sns.xkcd_rgb['greyish'],
        width=.4
        )

    boxplot(
        ax, xx,
        mean=distw.median('sample')[-1],
        box=(distw.median('sample')[1], distw.median('sample')[3]),
        whis=(distw.median('sample')[0], distw.median('sample')[4]),
        color=color1,
        width=.3
        )

    for idx, gs in dist.groupby('quantile'):
        if idx == 2:  # don't use median for now
            continue

        boxplot(
            ax, xx+.32,
            data=gs,
            showmedian=False,
            showmean=False,
            showdots=False,
            color=sns.xkcd_rgb['greyish'],
            width=.2,
            zorder=2,
            whis_kwargs={'lw': .5},
        )

    for idx, gs in distw.groupby('quantile'):
        if idx == 2:  # don't use median for now
            continue

        boxplot(
            ax, xx+.32,
            data=gs,
            showmedian=False,
            showmean=False,
            showdots=False,
            color=color1,
            width=.1,
            zorder=3,
            whis_kwargs={'lw': .5},
            showcaps=False,
        )

    boxplot(
        ax, xx+1.1,
        data=ds['tas_mean'],
        showmedian=False,
        showdots=False,
        box_quantiles=(1/6., 5/6.),
        color=sns.xkcd_rgb['greyish'],
        zorder=2,
    )

    boxplot(
        ax, xx+1.1,
        data=ds['tas_mean'],
        showmedian=False,
        showdots=False,
        box_quantiles=(1/6., 5/6.),
        weights=ds['weights_mean'],
        width=.6,
        color=color,
        zorder=3,
    )


    # ax.hlines(quantile(ds['tas_mean'], (.05, .95)),
    #           xx+.4, xx+1.1, color=sns.xkcd_rgb['greyish'], ls='--', lw=.5, zorder=.5)

    # ax.hlines([
    #     np.average(ds['tas_mean']),
    #     *quantile(ds['tas_mean'], (1/6., 5/6.))],
    #           xx+.4, xx+.7, color=sns.xkcd_rgb['greyish'], ls='--', lw=.5, zorder=.5)


    # ax.hlines(quantile(ds['tas_mean'], (.05, .95), weights=ds['weights_mean']),
    #           xx+.37, xx+1.1, color=color, ls='--', lw=.5, zorder=99)

    # ax.hlines([
    #     np.average(ds['tas_mean'], weights=ds['weights_mean']),
    #     *quantile(ds['tas_mean'], (1/6., 5/6.), weights=ds['weights_mean'])],
    #           xx+.37, xx+.8, color=color, ls='--', lw=.5, zorder=99)


def plot_weights(ds, ds_bt):
    """Lineplot of weights per model."""

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=.25, top=.95, right=.99, left=.08)

    xx = np.arange(ds.dims['model'])
    weights = ds['weights_mean'].data
    sorter = np.argsort(weights)[::-1]

    # making sure its normalized
    weights /= weights.sum()
    ds_bt = ds_bt.assign(weights = lambda xx: xx.weights / xx.weights.sum('model'))

    weights_bt_mean = ds_bt['weights'].mean('sample').data
    weights_bt_16 = ds_bt['weights'].load().quantile(1/6., 'sample').data
    weights_bt_83 = ds_bt['weights'].load().quantile(5/6., 'sample').data
    weights_bt_05 = ds_bt['weights'].load().quantile(.05, 'sample').data
    weights_bt_95 = ds_bt['weights'].load().quantile(.95, 'sample').data


    for idx, idx_sort in enumerate(sorter):
        hh2 = boxplot(ax, idx,
                     mean=get_factor(weights_bt_mean)[idx_sort],
                     box=(get_factor(weights_bt_16)[idx_sort], get_factor(weights_bt_83)[idx_sort]),
                      whis=(get_factor(weights_bt_05)[idx_sort], get_factor(weights_bt_95)[idx_sort]),
                      zorder=1)


    [hh1] = ax.plot(xx, get_factor(weights)[sorter], lw=2, color='k', marker='o', zorder=99)
    #, label='All ensemble variants')
    # ax.plot(xx, get_factor(weights_bt_mean)[sorter], lw=2, color='k', marker='o', label='Bootstrap mean')
    # ax.fill_between(xx, get_factor(weights_bt_16)[sorter], get_factor(weights_bt_83)[sorter],
    #                 color='gray', alpha=.3)
    # ax.fill_between(xx, get_factor(weights_bt_05)[sorter], get_factor(weights_bt_95)[sorter],
    #                 color='gray', alpha=.3, label='Bootstrap 90% range')

    # ax.fill_between([], [], [], color='gray', alpha=.6, label='Bootstrap 66% range')

    ax.set_xticks(xx)
    xticklabels = [f'{mm} ({nr})' for mm, nr in zip(
        ds['model'].data[sorter], ds['variant_count_mean'].data[sorter])]
    ax.set_xticklabels(xticklabels, rotation=60, ha='right')

    ax.set_xlim(-.5, xx.max()+.5)
    ax.set_ylim(-5, 5)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_yticklabels(['x0.16', 'x0.2', 'x0.25', 'x0.33', 'x0.5', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
    ax.set_ylabel('Factor (1)')

    ax.grid(axis='y')
    ax.legend([hh1, hh2], ['All ensemble variants', 'Bootstrap: mean, 66%, 90%'])

    ax.set_title('(b) Weights per model and independence scaling', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    filename = os.path.join('weights_bt.png')
    plt.savefig(os.path.join('figures/', filename), dpi=300)
    SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
    plt.savefig(os.path.join(SAVEPATH2, filename.replace('.png', '.pdf')), dpi=300)
    plt.close()



def main():
    ds_bt = load_bootstrap()
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp126_050_41-60.nc')
    ds = ds.assign(model = lambda xx: [x.split('_')[0] for x in xx.model.data])
    ds, ds_bt = sort_by_model(ds, ds_bt)

    plot_weights(ds, ds_bt)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=.05, right=.99, bottom=.06, top=.93)
    colors = sns.color_palette('colorblind', 4)
    colors1 = sns.color_palette('colorblind', 4, desat=.5)
    colors = np.array(colors)[np.array([0, 3])]
    colors1 = np.array(colors1)[np.array([0, 3])]

    plot_distributions(ax, 0, ds, ds_bt, color=colors[0], color1=colors1[0])

    # SSP585, 41-60
    ds_bt = load_bootstrap(ssp='ssp585')
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp585_050_41-60.nc')
    ds = ds.assign(model = lambda xx: [x.split('_')[0] for x in xx.model.data])
    ds, ds_bt = sort_by_model(ds, ds_bt)
    plot_distributions(ax, 2, ds, ds_bt, color=colors[1], color1=colors1[1])

    # SSP126, 81-00
    ds_bt = load_bootstrap(period='81-00')
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp126_050_81-00.nc')
    ds = ds.assign(model = lambda xx: [x.split('_')[0] for x in xx.model.data])
    ds, ds_bt = sort_by_model(ds, ds_bt)
    plot_distributions(ax, 4.5, ds, ds_bt, color=colors[0], color1=colors1[0])

    # SSP585, 81-00
    ds_bt = load_bootstrap(ssp='ssp585', period='81-00')
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp585_050_81-00.nc')
    ds = ds.assign(model = lambda xx: [x.split('_')[0] for x in xx.model.data])
    ds, ds_bt = sort_by_model(ds, ds_bt)
    plot_distributions(ax, 6.5, ds, ds_bt, color=colors[1], color1=colors1[1])

    ax.set_xticks([1.7, 6.2])
    ax.set_xticklabels(['2041-2060', '2081-2100'])

    hh1 = boxplot(ax, mean=[0], box=[0], color=sns.xkcd_rgb['greyish'], return_handle=True)
    hh2 = boxplot(ax, mean=[0], box=[0], color=colors[0], return_handle=True)
    hh3 = boxplot(ax, mean=[0], box=[0], color=colors[1], return_handle=True)
    hh4 = boxplot(ax, mean=[0], box=[0], color=colors1[0], return_handle=True)
    hh5 = boxplot(ax, mean=[0], box=[0], color=colors1[1], return_handle=True)

    leg = plt.legend([hh1, hh2, hh3, hh4, hh5],
                     ['Unweighted', 'SSP1-2.6 all ensemble members', 'SSP5-8.5 all ensemble members',
                      'SSP1-2.6 bootstrap', 'SSP5-8.5 bootstrap'],
                     loc='upper left', title='Mean, 66\%, 90\%')
    leg._legend_box.align = 'left'

    ax.set_ylabel('Temperature change (°C) relative to 1995-2014')
    ax.grid(axis='y')
    ax.set_ylim((0, 6))

    ax.set_title('\\textbf{CMIP6 global temperature change (full ensemble and bootstrap)}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    filename = os.path.join('figure8a_supp_bootstrap.png')
    plt.savefig(os.path.join('figures/', filename), dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure8a_supp_bootstrap.pdf'), dpi=300)
    plt.close()



if __name__ == '__main__':
    main()
