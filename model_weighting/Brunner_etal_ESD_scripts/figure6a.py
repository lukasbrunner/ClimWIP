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

SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'


def get_factor(arr):
    arr[arr < 1.] = 1 - 1/arr[arr < 1.]
    arr[arr >= 1.] -= 1
    return arr


def plot_weights(ds, ds2, pseudo_model):
    """Lineplot of weights per model.

    Parameters
    ----------
    ds : xarray.Dataset
        Has to contain the 'weights' variable (shape (N,)) depending on the
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
    fig.subplots_adjust(bottom=.2, top=.95, right=.93, left=.06)

    models = ds['model'].data
    models = [model.split('_')[0] for model in models]
    idx_org = np.where(np.array(models) == pseudo_model)[0]
    weights_q = ds['weights_q_mean'].data
    weights_i = ds['weights_i_mean'].data
    weights = weights_q/weights_i
    weights /= weights.sum()

    models2 = ds2['model'].data
    models2 = [model.split('_')[0] for model in models2]
    idx_pm = np.where(np.array(models2) == pseudo_model + '-COPY')[0]
    weights_i_pm = ds2['weights_i_mean'].data

    weight_temp = weights_i_pm[idx_pm]
    weights_i_pm = np.delete(weights_i_pm, idx_pm)
    weights_i_pm = np.insert(weights_i_pm, 0, weight_temp)
    weights_q_pm = np.insert(weights_q, 0, weights_q[idx_org])
    weights_pm = weights_q_pm/weights_i_pm
    weights_pm /= weights_pm.sum()
    weights_pm_0 = weights_pm[0]
    weights_pm = np.delete(weights_pm, 0)

    xx = np.arange(len(weights))
    sorter = np.argsort(weights)[::-1].data

    change = ((weights_pm -weights) / weights)[sorter]
    # print(np.sum(change))
    yy1 = weights[sorter]
    yy2 = weights_pm[sorter]
    yy3 = weights_pm_0

    model_ensemble = np.delete(ds2['model'].data, idx_pm)
    # model_ensemble = ds2['model'].data
    model_ensemble = np.array([me.split('_')[0] for me in model_ensemble])[sorter]

    ax2 = ax.twinx()
    hh = ax2.bar(xx, change, zorder=10)
    handles = [hh]
    labels = ['Change in combined weight (right axis)']
    scale = 8.34
    ax2.set_ylim(-.125*scale, .015*scale)
    ax2.set_yticks([-.5, -1/3, -1/6, 0])
    ax2.set_yticklabels(['-50\%', '-33\%', '-17\%', '$\pm 0\%$'])
    ax2.set_ylabel('Weight change (\%)')
    ax2.yaxis.set_label_coords(1.06, .68)

    hh = ax.axhline(1/len(xx), color='k', zorder=2, ls='--')
    handles.append(hh)
    labels.append('Equal weighting')

    [hh] = ax.plot(xx, yy1, lw=2, color='gray', label='Original combined weight', marker='o', zorder=20)
    handles.append(hh)
    labels.append('Original combined weight')
    [hh] = ax.plot(xx, yy2, lw=2, color='k',
            label=f'Original performance; independence with\n{pseudo_model}-r1i1p1f1 as separate model',
            marker='o', zorder=20)
    handles.append(hh)
    labels.append(f'Original performance; independence with\n{pseudo_model}-r1i1p1f1 as separate model')

    ax.set_xticks(xx)
    ax.set_xticklabels([
        '%s%s%s' % ('\\textbf{', mm, '}')
        if mm in ['MIROC6', 'MIROC-ES2L']
        else mm
        for mm in  model_ensemble], rotation=60, ha='right')
    ax.set_xlim(-0.5, xx.max()+.5)

    ax.set_ylim(-.005, .135)
    ax.set_ylabel('Weight (1)')

    ax.grid(axis='y')

    ax.legend(handles, labels, loc=(.665, .19))

    ax.set_title(
        '\\textbf{(a) Weights per model, MIROC6-r1i1p1f1 as separate model}',
        loc='left',
        fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join('figures/', 'figure6a.png'), dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure6a.pdf'), dpi=300)
    plt.close()


def main():
    pseudo_model = 'MIROC6'
    ds = xr.open_dataset(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1/tas_global_ssp585_050_81-00.nc')
    ds_pm = xr.open_dataset(os.path.join(
        '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1/',
        f'tas_global_ssp585_050_81-00_{pseudo_model}.nc'))
    plot_weights(ds, ds_pm, pseudo_model)


if __name__ == '__main__':
    main()
