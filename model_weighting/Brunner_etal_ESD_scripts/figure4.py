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

dict_tcr = {
    'ACCESS-CM2': 2.11,
    'ACCESS-ESM1-5': 1.95,
    'AWI-CM-1-1-MR': 2.07,
    'BCC-CSM2-MR': 1.50,
    'BCC-ESM1': 1.73,
    'CAMS-CSM1-0': 1.75,
    'CAS-ESM2-0': 2.12,
    'CanESM5': 2.66,
    'CanESM5-CanOE': 2.64,
    'CESM2': 2.06,
    'CESM2-FV2': 2.05,
    'CESM2-WACCM': 1.98,
    'CESM2-WACCM-FV2': 2.01,
    'CIESM': 2.39,
    'CMCC-CM2-SR5': 2.08,
    'CNRM-CM6-1': 2.13,
    'CNRM-CM6-1-HR': 2.47,
    'CNRM-ESM2-1': 1.92,
    'E3SM-1-0': 2.99,
    'EC-Earth3': 2.49,
    'EC-Earth3-Veg': 2.61,
    'FGOALS-f3-L': 2.06,
    'FGOALS-g3': 1.57,
    'FIO-ESM-2-0': 2.24,
    'GFDL-CM4': 2.01,
    'GFDL-ESM4': 1.61,
    'GISS-E2-1-G': 1.80,
    'GISS-E2-2-G': 1.72,
    'GISS-E2-1-H': 1.92,
    'HadGEM3-GC31-LL': 2.51,
    'HadGEM3-GC31-MM': 2.58,
    'IITM-ESM': 1.70,
    'INM-CM4-8': 1.32,
    'INM-CM5-0': 1.39,
    'IPSL-CM6A-LR': 2.31,
    'KACE-1-0-G': 2.19,
    'MCM-UA-1-0': 1.94,
    'MIROC6': 1.55,
    'MIROC-ES2L': 1.55,
    'MPI-ESM-1-2-HAM': 1.81,
    'MPI-ESM1-2-HR': 1.65,
    'MPI-ESM1-2-LR': 1.84,
    'MRI-ESM2-0': 1.65,
    'NESM3': 2.79,
    'NorCPM1': 1.56,
    'NorESM2-LM': 1.48,
    'NorESM2-MM': 1.34,
    'SAM0-UNICON': 2.26,
    'TaiESM1': 2.35,
    'UKESM1-0-LL': 2.75,
}

def label_font_color_tcr(lbls):
    colors = sns.color_palette('colorblind', 4)
    for lbl in lbls:
        model = lbl.get_text().lstrip('\\textbf{').split()[0]
        # lbl.set_text('%s%s%s' % ('\\textbf{', lbl.get_text(), '}'))
        tcr = dict_tcr[model]
        if tcr is None:
            continue
        elif tcr > 2.5:
            lbl.set_color(colors[3])
        elif tcr > 2.:
            lbl.set_color(colors[1])
        elif tcr > 1.5:
            lbl.set_color(colors[2])
        else:
            lbl.set_color(colors[0])


def plot_weights(ds):
    """Lineplot of weights per model.

    Parameters
    ----------
    ds : xarray.Dataset
        Has to contain the 'weihgts' variable (shape (N,)) depending on the
        'model_ensemble' dimension.
    cfg : model_weighting.config object
    nn : array_like, shape (N,)
        Array of performance weights (numerator in weights equation)
    dd : array_like, shape (N,)
        Array of independence weights (denominator in weights equation)
    sort_by : string, optional
        name: plot models alphabetically and cluster ensemble members.
        weights: plot models by weight (highest first)
        performance: plot models by performance weight (highest first)
        independence: plot models by independence weight (highest first)
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=.26, top=.98, right=.99, left=.06)
    handles = []
    labels = []

    model_ensemble = ds['model'].data
    xx = np.arange(ds.dims['model'])
    dd = 1/ds['weights_i_mean'].data
    nn = ds['weights_q_mean'].data
    ww = nn*dd

    sorter = np.argsort(ww)[::-1]

    yy1 = (ww/ww.sum())[sorter]
    yy2 = (nn/nn.sum())[sorter]
    yy3 = (dd/dd.sum())[sorter]

    tt = yy2*yy3
    tt /= tt.sum()

    hh = ax.axhline(1/len(xx), color='k', zorder=2, ls='--')
    handles.append(hh)
    labels.append('Equal weighting')

    [hh] = ax.plot([], [], lw=2, color='k', marker='o', ms=4)
    handles.append(hh)
    labels.append('Combined weight')

    ax.plot(xx, yy1, lw=2, color='k',
            zorder=20)
    ax.scatter(xx, yy1, c='k',  # get_colors(model_ensemble[sorter])
               s=40, zorder=25)
    [hh] = ax.plot(xx, yy2, lw=2, color='gray',
            marker='s', ms=3, ls='none', zorder=30)
    handles.append(hh)
    labels.append('Performance weight')

    [hh] = ax.plot(xx, yy3, lw=2, color='gray',
            marker='^', ms=3, ls='none', zorder=30)
    handles.append(hh)
    labels.append('Independence weight')

    model_ensemble = np.array(
        [me.split('_')[0] + ' (' + (me.split('_')[1] if len(me.split('_')[1]) < 3 else '1') + ')'
         for me in model_ensemble])
    ax.set_xticks(xx)
    ax.set_xticklabels(['%s%s%s' % ('\\textbf{', mm, '}') for mm in model_ensemble[sorter]], rotation=60, ha='right')
    label_font_color_tcr(ax.get_xmajorticklabels())

    ax.set_xlim(-.5, xx.max()+.5)
    ax.set_ylabel('Weight (1)')
    ax.set_ylim(-.005, .135)

    ax.grid(axis='y')
    ax.legend(handles, labels, ncol=4)

    # ax.set_title('Weights per model and independence scaling', loc='left',
    #              fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join('figures/', 'figure4.png'), dpi=300)
    SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
    plt.savefig(os.path.join(SAVEPATH2, 'figure4.pdf'), dpi=300)
    plt.close()



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
            line = ' & '.join(
                [labels2[idx2], *['{:.2f}'.format(l) if not isinstance(l, tuple)
                                  else '{:.2f} - {:.2f}'.format(*l) for l in ll]])
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


def print_tcr_weights(weights):
    def color_tcr(value):
        colors = sns.color_palette('colorblind', 4)
        if value > 2.5:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[3]]) + '}' + str(value)
        elif value > 2.:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[2]]) + '}' + str(value)
        elif value > 1.5:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[1]]) + '}' + str(value)
        else:
            return '\\cellcolor[rgb]{' + ','.join([f'{cc:.4f}' for cc in  colors[0]]) + '}' + str(value)

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


    model_ensembles = weights['model'].data

    lines = []
    lines += ['\\begin{tabular}{llll}']
    lines += [' & '.join(['Model', 'TCR', 'Weight', 'Predecessor']) + '\\\\ \\hline']
    for model_ensemble in model_ensembles:
        model = model_ensemble.split('_')[0]
        line = ' & '.join([
            model,
            color_tcr(dict_tcr[model]),  # str(dict_tcr[model]),
            color_weights(weights.sel(model=model_ensemble).data),
            ''])
        line += ' \\\\'
        lines.append(line)

    lines += ['\\end{tabular}']
    SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/'
    with open(os.path.join(SAVEPATH2, 'table_tcr_weighs.tex'), 'w') as ff:
        ff.write('\n'.join(lines))


def main():
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp585_050_81-00.nc')
    # print_tcr_weights(ds['weights_mean'])
    plot_weights(ds)


if __name__ == '__main__':
    main()
