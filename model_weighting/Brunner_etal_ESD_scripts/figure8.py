#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-09-16 09:16:02 lukas>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from adjustText import adjust_text
import natsort as ns

from model_weighting.core.utils_xarray import area_weighted_mean, quantile
from boxplot import boxplot

SAVEPATH = './figures'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FILENAMES = [
    'tas_global_ssp126_050_41-60.nc',
    'tas_global_ssp585_050_41-60.nc',
    '',
    'tas_global_ssp126_050_81-00.nc',
    'tas_global_ssp585_050_81-00.nc',
]


tcrs = {
    'ACCESS-CM2': '2.11',
    'ACCESS-ESM1-5': '1.95',
    'AWI-CM-1-1-MR': '2.07',
    'BCC-CSM2-MR': '1.50',
    'BCC-ESM1': '1.73',
    'CAMS-CSM1-0': '1.75',
    'CAS-ESM2-0': '2.12',
    'CanESM5': '2.66',
    'CanESM5-CanOE': '2.64',
    'CESM2': '2.06',
    'CESM2-FV2': '2.05',
    'CESM2-WACCM': '1.98',
    'CESM2-WACCM-FV2': '2.01',
    'CIESM': '2.39',
    'CMCC-CM2-SR5': '2.08',
    'CNRM-CM6-1': '2.13',
    'CNRM-CM6-1-HR': '2.47',
    'CNRM-ESM2-1': '1.92',
    'E3SM-1-0': '2.99',
    'EC-Earth3': '2.49',
    'EC-Earth3-Veg': '2.61',
    'FGOALS-f3-L': '2.06',
    'FGOALS-g3': '1.57',
    'FIO-ESM-2-0': '2.24',
    'GFDL-CM4': '2.01',
    'GFDL-ESM4': '1.61',
    'GISS-E2-1-G': '1.80',
    'GISS-E2-2-G': '1.72',
    'GISS-E2-1-H': '1.92',
    'HadGEM3-GC31-LL': '2.51',
    'HadGEM3-GC31-MM': '2.58',
    'IITM-ESM': '1.70',
    'INM-CM4-8': '1.32',
    'INM-CM5-0': '1.39',
    'IPSL-CM6A-LR': '2.31',
    'KACE-1-0-G': '2.19',
    'MCM-UA-1-0': '1.94',
    'MIROC6': '1.55',
    'MIROC-ES2L': '1.55',
    'MPI-ESM-1-2-HAM': '1.81',
    'MPI-ESM1-2-HR': '1.65',
    'MPI-ESM1-2-LR': '1.84',
    'MRI-ESM2-0': '1.65',
    'NESM3': '2.79',
    'NorCPM1': '1.56',
    'NorESM2-LM': '1.48',
    'NorESM2-MM': '1.34',
    'SAM0-UNICON': '2.26',
    'TaiESM1': '2.35',
    'UKESM1-0-LL': '2.75'}




def read_data(filename, path):
    ds = xr.open_dataset(os.path.join(path, filename))
    ds = area_weighted_mean(ds)
    varn = ds.attrs['target']
    return ds


def get_statistics_tcr(data, weights):
    mean = np.mean(data)
    median = np.median(data)
    range66 = (quantile(data, 1/6.), quantile(data, 5/6.))
    range90 = (quantile(data, .05), quantile(data, .95))

    mean_w = np.average(data, weights=weights)
    median_w = quantile(data, .5, weights=weights)
    range66_w = (quantile(data, 1/6., weights=weights),
                 quantile(data, 5/6., weights=weights))
    range90_w = (quantile(data, .05, weights=weights),
                 quantile(data, .95, weights=weights))

    mean_c = np.around(mean_w, 2) - np.around(mean, 2)
    median_c = np.around(median_w, 2) - np.around(median, 2)
    range66_c = ((np.around(range66_w[1] - range66_w[0], 2) -
                 np.around(range66[1] - range66[0], 2)) / np.around(range66[1] - range66[0], 2) * 100)
    range90_c = ((np.around(range90_w[1] - range90_w[0], 2) -
                 np.around(range90[1] - range90[0], 2)) / np.around(range90[1] - range90[0], 2) * 100)

    return [
        [mean,
         median,
         range66,
         range90],
        [mean_w,
         median_w,
         range66_w,
         range90_w],
        [mean_c,
         median_c,
         range66_c,
         range90_c]]


def get_statistics(ds, varn):
    mean = ds[varn].mean().data
    median = ds[varn].median().data
    range66 = (quantile(ds[varn], 1/6.), quantile(ds[varn], 5/6.))
    range90 = (quantile(ds[varn], .05), quantile(ds[varn], .95))

    mean_w = np.average(ds[varn], weights=ds['weights'])
    median_w = quantile(ds[varn], .5, weights=ds['weights'])
    range66_w = (quantile(ds[varn], 1/6., weights=ds['weights']),
                 quantile(ds[varn], 5/6., weights=ds['weights']))
    range90_w = (quantile(ds[varn], .05, weights=ds['weights']),
                 quantile(ds[varn], .95, weights=ds['weights']))

    mean_c = np.around(mean_w, 2) - np.around(mean, 2)
    median_c = np.around(median_w, 2) - np.around(median, 2)
    range66_c = ((np.around(range66_w[1] - range66_w[0], 2) -
                 np.around(range66[1] - range66[0], 2)) / np.around(range66[1] - range66[0], 2) * 100)
    range90_c = ((np.around(range90_w[1] - range90_w[0], 2) -
                 np.around(range90[1] - range90[0], 2)) / np.around(range90[1] - range90[0], 2) * 100)

    return [
        [mean,
         median,
         range66,
         range90],
        [mean_w,
         median_w,
         range66_w,
         range90_w],
        [mean_c,
         median_c,
         range66_c,
         range90_c]]


def print_tcr_weights(weights, warming):
    def color_tcr(value):
        colors = sns.color_palette('colorblind', 4)
        value = float(value)
        if value > 2.5:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[3]]) + '}' + '\\SI{' + str(value) + '}{\\degree C}'
        elif value > 2.:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[1]]) + '}' + '\\SI{' + str(value) + '}{\\degree C}'
        elif value > 1.5:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[2]]) + '}' + '\\SI{' + str(value) + '}{\\degree C}'
        else:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[0]]) + '}' + '\\SI{' + str(value) + '}{\\degree C}'

    def color_weights(value):
        value_relative = value * len(weights)
        value_relative =  1 - 1/value_relative if value_relative < 1. else value_relative - 1
        colors = sns.color_palette('vlag', 11)
        bounds = [-2.5, -2, -1.5, -1, -.5, .5, 1, 1.5, 2, 2.5]
        for idx, bound in enumerate(bounds[::-1]):
            if value_relative > bound:
                return '\\cellcolor[rgb]{' + ','.join(
                    [f'{cc:.4f}' for cc in  colors[::-1][idx]]) + '}' + '{:.4f}'.format(value)
        return '\\cellcolor[rgb]{' + ','.join(
            [f'{cc:.4f}' for cc in  colors[0]]) + '}' + '{:.4f}'.format(value)


    model_ensembles = ns.natsorted(weights['model_ensemble'].data, alg=ns.IC)

    lines = []
    lines += ['\\begin{tabular}{lll | llll}']
    lines += ['&&& \\multicolumn{2}{c}{2041-2060} & \multicolumn{2}{c}{2081-2100} \\\\']
    lines += [' & '.join(['Model', 'Weight', 'TCR', 'SSP1-2.6', 'SSP5-8.5', 'SSP1-2.6', 'SSP5-8.5']) + ' \\\\ \\hline']
    for model_ensemble in model_ensembles:
        model = model_ensemble.split('_')[0]
        line = ' & '.join([
            model,
            color_weights(weights.sel(model_ensemble=model_ensemble).data),
            color_tcr(tcrs[model]),
            '\\SI{' + '{:.2f}'.format(warming[0][model_ensemble]) + '}{\\degree C}',
            '\\SI{' + '{:.2f}'.format(warming[1][model_ensemble]) + '}{\\degree C}',
            '\\SI{' + '{:.2f}'.format(warming[2][model_ensemble]) + '}{\\degree C}',
            '\\SI{' + '{:.2f}'.format(warming[3][model_ensemble]) + '}{\\degree C}',
        ])
        line += ' \\\\'
        lines.append(line)

    lines += ['\\end{tabular}']
    SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/'
    with open(os.path.join(SAVEPATH2, 'table_tcr_weighs.tex'), 'w') as ff:
        ff.write('\n'.join(lines))



def print_statistics_table(list_nested):
    labels = ['SSP1-2.6 2041-2060', 'SSP5-8.5 2041-2060', 'SSP1-2.6 2081-2100', 'SSP5-8.5 2081-2100', 'TCR']
    labels2 = ['Unweighted', 'Weighted', 'Change']

    lines = []
    for idx, lists in enumerate(list_nested):
        line = '\\begin{tabular}{l | llll}'
        lines.append(line)

        line = ' & '.join([labels[idx], 'Mean', 'Median', '66\,\% range', '90\,\% range'])
        line += ' \\\\ \\hline'
        lines.append(line)

        for idx2, ll in enumerate(lists):
            if idx2 in [0, 1]:
                unit = ['\\degree C'] * 4
            else:
                unit = ['\\degree C'] * 2 + ['\\percent'] * 2
            line = ' & '.join(
                [labels2[idx2], *[
                    '\\SI{' + '{:.2f}'.format(l) + '}{' + unit[idx3] + '}' if not isinstance(l, tuple)
                    else '\\SIrange{' + '{:.2f}'.format(l[0]) + '}{' +  '{:.2f}'.format(l[1])  + '}{\\degree C}'
                    for idx3, l in enumerate(ll)]])
            line += ' \\\\'
            if idx2 == 1:
                line += '[2mm]'
            lines.append(line)

        line = '\\end{tabular}'
        lines.append(line)
        if idx != 4:
            lines.append('')
            line = '\\vspace*{5mm}'
            lines.append(line)
            lines.append('')

    with open(os.path.join(SAVEPATH2, '../table_tas_change.tex'), 'w') as ff:
        ff.write('\n'.join(lines))

def plot_tcr(ax, weights):
    model_ensemble = weights['model_ensemble'].data
    models = [me.split('_')[0] for me in model_ensemble]
    weights = weights.assign_coords({'model_ensemble': models})

    colors = sns.color_palette('colorblind', 4)

    # # get subset of TCR for which I have weights:
    # idx = np.array([idx for idx, model in enumerate(models) if model in weights['model_ensemble'].data])
    # models = np.array(models)[idx]
    # tcrs = np.array(tcrs)[idx]

    models_tcr_missing = [key for key in models if key not in tcrs.keys()]
    assert len(models_tcr_missing) == 0

    tcr_data = [float(tcrs[key]) for key in models]
    weights_data = weights.sel(model_ensemble=models).data
    # import ipdb; ipdb.set_trace()

    # idx_tcr = np.array([idx for idx in np.arange(len(models))
    #                     if models[idx] in weights['model_ensemble'].data and tcrs[idx] != ''])
    # models_tcr = np.array(models)[idx_tcr]
    # tcrs = [*map(float, np.array(tcrs)[idx_tcr])]


    # idx_sel_tcr = [idx for idx, model in enumerate(weights['model_ensemble']) if model in models_tcr]
    # weights_tcr = weights.isel(model_ensemble=idx_sel_tcr).sel(model_ensemble=models_tcr)

    boxplot(ax, data=tcr_data, showdots=False, showmean=True, showmedian=False, box_quantiles=(1/6., 5/6.))
    hh =boxplot(ax, data=tcr_data, weights=weights_data, showdots=False, showmean=True, showmedian=False,
            width=.6, color=colors[2], dots_sizes=(1, 5), box_quantiles=(1/6., 5/6.))

    xx = np.random.RandomState(0).uniform(-.4*.6, .4*.6, len(models))
    sizes = np.interp(weights_data, [np.quantile(weights_data, .1), np.quantile(weights_data, .9)], (1, 10))
    texts = []
    dots = []

    label_list = [
        [0.04, 2.11, "ACCESS-CM2"], #
        [0.12, 1.92, "ACCESS-ESM1-5"],
        [-0.36, 2.1, "AWI-CM-1-1-MR"], #
        [0.03, 1.47, "BCC-CSM2-MR"],
        [-0.02, 1.72, "CAMS-CSM1-0"],
        [0.08, 1.98, "CESM2-WACCM"],
        [-0.02, 2.04, "CESM2"],
        [0.19, 2.43, "CNRM-CM6-1-HR"], #
        [0.24, 2.12, "CNRM-CM6-1"], #
        [-0.14, 1.94, "CNRM-ESM2-1"],
        [0.15, 2.64, "CanESM5-CanOE"], #
        [0.02, 2.66, "CanESM5"], #
        [0.02, 2.61, "EC-Earth3-Veg", {'ha': 'right'}],
        [-0.09, 2.49, "EC-Earth3"], #
        [-0.19, 2.01, "FGOALS-f3-L"],
        [-0.2, 1.57, "FGOALS-g3", {'ha': 'right'}],
        [-0.21, 2.24, "FIO-ESM-2-0"], #
        [0.17, 1.56, "GFDL-ESM4"],
        [0.15, 1.80, "GISS-E2-1-G"],
        [0.19, 2.51, "HadGEM3-GC31-LL"], #
        [0.24, 1.32, "INM-CM4-8"],
        [0.17, 1.39, "INM-CM5-0"],
        [-0.0, 2.31, "IPSL-CM6A-LR"], #
        [0.15, 2.19, "KACE-1-0-G"], #
        [-0.33, 1.94, "MCM-UA-1-0"],
        [0.08, 1.53, "MIROC6"],
        [-0.16, 1.55, "MIROC-ES2L"],
        [0.23, 1.66, "MPI-ESM1-2-HR"],
        [-0.01, 1.84, "MPI-ESM1-2-LR", {'ha': 'right'}],#
        [-0.02, 1.65, "MRI-ESM2-0"],
        [-0.10, 2.79, "NESM3"], #
        [0.12, 1.32, "NorESM2-MM", {'ha': 'right'}],
        [-0.01, 2.75, "UKESM1-0-LL"], #
    ]

    for idx, model in enumerate(models):
        if model == 'AWI-CM-1-1-MR':
            xx[idx] = -0.37
        if model == 'EC-Earth3':
            xx[idx] = -.10
        if model == 'MCM-UA-1-0':
            xx[idx] = -.35
        # print(f'[{xx[idx]:.2f}, {tcrs[model]}, "{model}"],')
        [dot] = ax.plot(xx[idx], float(tcrs[model]), color='k', ms=sizes[idx], marker='o', zorder=9999)
        ax.text(*label_list[idx], color='k', zorder=999, size='x-small')
        # dots.append(dot)
        # texts.append(ax.text(xx[idx], float(tcrs[model]), model, color='k', zorder=999, size='x-small'))
    # adjust_text(texts, add_objects=dots)

    ax.set_ylabel('Transient Climate Response ($^\circ$C)')
    ax.set_xticks([])
    ax.set_ylim(1, 3.5)
    ax.set_title('\\textbf{(b) CMIP6 Transent Climate Responce}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    return hh, get_statistics_tcr(tcr_data, weights_data)


def main():

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.subplots_adjust(left=.05, right=.99, bottom=.05, top=.94)
    colors = sns.color_palette('colorblind', 4)
    colors = np.array(colors)[np.array([0, 3])]

    ax = axs[0]

    statistics = []
    warming = []
    for xx, filename in enumerate(FILENAMES):
        if filename == '':
            continue

        ds = read_data(filename, LOADPATH)
        varn = ds.attrs['target']

        if 'weights_mean' in ds:
            ds = ds.drop_dims(('model_ensemble', 'perfect_model_ensemble'))
            ds = ds.rename({'model': 'model_ensemble', 'perfect_model': 'perfect_model_ensemble',
                            f'{varn}_mean': varn, 'weights_mean': 'weights'})

        statistics.append(get_statistics(ds, varn))

        h1 = boxplot(
            ax, xx,
            data=ds[varn],
            showmean=True,
            showmedian=False,
            showdots=False,
            box_quantiles=(1/6., 5/6.),
            color=sns.xkcd_rgb['greyish'],
        )

        boxplot(
            ax, xx,
            data=ds[varn],
            showmean=True,
            showmedian=False,
            # showdots=False,
            dots_sizes=(1, 5),
            box_quantiles=(1/6., 5/6.),
            weights=ds['weights'],
            width=.6,
            color=colors[0] if 'ssp126' in filename else colors[1],
        )

        dd = {}
        for model_ensemble in ds['model_ensemble'].data:
            dd[model_ensemble] = ds.sel(model_ensemble=model_ensemble)[varn].data
        warming.append(dd)

    print_tcr_weights(ds['weights'], warming)

    ax.set_xticks([.5, 3.5])
    ax.set_xticklabels(['2041-2060', '2081-2100'])

    ax.set_ylabel('Temperature change (°C) relative to 1995-2014')
    ax.grid(axis='y')

    ax.set_ylim((0, 6))

    ds = read_data('tas_global_ssp126_050_41-60.nc', LOADPATH)
    varn = ds.attrs['target']
    if 'weights_mean' in ds:
        ds = ds.drop_dims(('model_ensemble', 'perfect_model_ensemble'))
        ds = ds.rename({'model': 'model_ensemble', 'perfect_model': 'perfect_model_ensemble',
                        f'{varn}_mean': varn, 'weights_mean': 'weights'})

    h4, data = plot_tcr(axs[1], ds['weights'])
    statistics.append(data)
    print_statistics_table(statistics)


    h2 = boxplot(ax, [], data=ds[varn], return_handle=True, showmean=False, showdots=False, color=colors[0])
    h3 = boxplot(ax, [], data=ds[varn], return_handle=True, showmean=False, showdots=False, color=colors[1])
    leg = ax.legend((h1, h2, h3, h4), ('Unweighted', 'SSP1-2.6 weighted', 'SSP5-8.5 weighted', 'TCR weighted'),
                    loc='upper left',
                    title='Mean, 66\%, 90\%')
    leg._legend_box.align = 'left'

    ax.set_title('\\textbf{(a) CMIP6 global temperature change}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join(SAVEPATH, 'figure8.png'), dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure8.pdf'), dpi=300)


if __name__ == '__main__':
    main()
