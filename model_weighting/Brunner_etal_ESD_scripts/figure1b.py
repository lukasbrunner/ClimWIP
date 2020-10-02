#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-09-10 19:26:41 lukas>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Calculate the CRPS of the target variable for the future period
based on a perfect model test. This plots the CRPS of the area weighted mean!
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
import seaborn as sns

from utils_python.xarray import area_weighted_mean

from boxplot import boxplot

SAVEPATH = '/home/lukbrunn/Documents/Scripts/climWIP_clean_paper/model_weighting/scripts_paper/revision1/figures'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
SAVENAME = 'figure1b.png'
FILENAME_PATTERN = 'tas_global_perfect_free_sigma_{ssp}_{trend}_81-00.nc'

FILENAMES126 = [
    FILENAME_PATTERN.format(ssp='ssp126', trend='000'),
    FILENAME_PATTERN.format(ssp='ssp126', trend='033'),
    FILENAME_PATTERN.format(ssp='ssp126', trend='050'),
    FILENAME_PATTERN.format(ssp='ssp126', trend='066'),
    FILENAME_PATTERN.format(ssp='ssp126', trend='100'),
]

FILENAMES585 = [
    FILENAME_PATTERN.format(ssp='ssp585', trend='000'),
    FILENAME_PATTERN.format(ssp='ssp585', trend='033'),
    FILENAME_PATTERN.format(ssp='ssp585', trend='050'),
    FILENAME_PATTERN.format(ssp='ssp585', trend='066'),
    FILENAME_PATTERN.format(ssp='ssp585', trend='100'),
]

# larger families
# exclude_models = {
#     'ACCESS-CM2_r1i1p1f1_CMIP6': [
#         'ACCESS-ESM1-5_3_CMIP6', 'UKESM1-0-LL_5_CMIP6', 'HadGEM3-GC31-LL_r1i1p1f3_CMIP6'],
#     'ACCESS-ESM1-5_3_CMIP6': [
#         'ACCESS-CM2_r1i1p1f1_CMIP6', 'UKESM1-0-LL_5_CMIP6', 'HadGEM3-GC31-LL_r1i1p1f3_CMIP6'],
#     'AWI-CM-1-1-MR_r1i1p1f1_CMIP6': [
#         'MPI-ESM1-2-HR_2_CMIP6', 'CAMS-CSM1-0_2_CMIP6', 'MPI-ESM1-2-LR_10_CMIP6', 'NESM3_2_CMIP6'],
#     'BCC-CSM2-MR_r1i1p1f1_CMIP6': [
#         'CESM2-WACCM_r1i1p1f1_CMIP6', 'CESM2_2_CMIP6', 'NorESM2-MM_r1i1p1f1_CMIP6'],
#     'CAMS-CSM1-0_2_CMIP6': [
#         'MPI-ESM1-2-HR_2_CMIP6', 'MPI-ESM1-2-LR_10_CMIP6', 'NESM3_2_CMIP6'],
#     'CESM2-WACCM_r1i1p1f1_CMIP6': [
#         'BCC-CSM2-MR_r1i1p1f1_CMIP6', 'CESM2_2_CMIP6', 'NorESM2-MM_r1i1p1f1_CMIP6'],
#     'CESM2_2_CMIP6': [
#         'BCC-CSM2-MR_r1i1p1f1_CMIP6', 'CESM2-WACCM_r1i1p1f1_CMIP6', 'NorESM2-MM_r1i1p1f1_CMIP6'],
#     'CNRM-CM6-1-HR_r1i1p1f2_CMIP6': [
#         'CNRM-CM6-1_6_CMIP6', 'CNRM-ESM2-1_5_CMIP6'],
#     'CNRM-CM6-1_6_CMIP6': [
#         'CNRM-CM6-1-HR_r1i1p1f2_CMIP6', 'CNRM-ESM2-1_5_CMIP6'],
#     'CNRM-ESM2-1_5_CMIP6': [
#         'CNRM-CM6-1-HR_r1i1p1f2_CMIP6', 'CNRM-CM6-1_6_CMIP6'],
#     'CanESM5-CanOE_3_CMIP6': [
#         'CanESM5_50_CMIP6'],
#     'CanESM5_50_CMIP6': [
#         'CanESM5-CanOE_3_CMIP6'],
#     'EC-Earth3-Veg_3_CMIP6': [
#         'EC-Earth3_7_CMIP6'],
#     'EC-Earth3_7_CMIP6': [
#         'EC-Earth3-Veg_3_CMIP6'],
#     'FGOALS-f3-L_r1i1p1f1_CMIP6': [
#         'FGOALS-g3_r1i1p1f1_CMIP6'],
#     'FGOALS-g3_r1i1p1f1_CMIP6': [
#         'FGOALS-f3-L_r1i1p1f1_CMIP6'],
#     'FIO-ESM-2-0_3_CMIP6': [],
#     'GFDL-ESM4_r1i1p1f1_CMIP6': [],
#     'GISS-E2-1-G_r1i1p3f1_CMIP6': [],
#     'HadGEM3-GC31-LL_r1i1p1f3_CMIP6': [
#         'UKESM1-0-LL_5_CMIP6', 'ACCESS-CM2_r1i1p1f1_CMIP6', 'ACCESS-ESM1-5_3_CMIP6'],
#     'INM-CM4-8_r1i1p1f1_CMIP6': [
#         'INM-CM5-0_r1i1p1f1_CMIP6'],
#     'INM-CM5-0_r1i1p1f1_CMIP6': [
#         'INM-CM4-8_r1i1p1f1_CMIP6'],
#     'IPSL-CM6A-LR_6_CMIP6': [
#         'MIROC6_3_CMIP6', 'MIROC-ES2L_r1i1p1f2_CMIP6'],
#     'KACE-1-0-G_r1i1p1f1_CMIP6': [],
#     'MCM-UA-1-0_r1i1p1f2_CMIP6': [],
#     'MIROC6_3_CMIP6': [
#         'IPSL-CM6A-LR_6_CMIP6', 'MIROC-ES2L_r1i1p1f2_CMIP6'],
#     'MIROC-ES2L_r1i1p1f2_CMIP6': [
#         'IPSL-CM6A-LR_6_CMIP6', 'MIROC6_3_CMIP6'],
#     'MPI-ESM1-2-HR_2_CMIP6': [
#         'CAMS-CSM1-0_2_CMIP6', 'MPI-ESM1-2-LR_10_CMIP6', 'NESM3_2_CMIP6', 'AWI-CM-1-1-MR_r1i1p1f1_CMIP6'],
#     'MPI-ESM1-2-LR_10_CMIP6': [
#         'CAMS-CSM1-0_2_CMIP6', 'MPI-ESM1-2-HR_2_CMIP6', 'NESM3_2_CMIP6', 'AWI-CM-1-1-MR_r1i1p1f1_CMIP6'],
#     'MRI-ESM2-0_r1i1p1f1_CMIP6': [],
#     'NESM3_2_CMIP6': [
#         'CAMS-CSM1-0_2_CMIP6', 'MPI-ESM1-2-HR_2_CMIP6', 'MPI-ESM1-2-LR_10_CMIP6'],
#     'NorESM2-MM_r1i1p1f1_CMIP6': [
#         'BCC-CSM2-MR_r1i1p1f1_CMIP6', 'CESM2-WACCM_r1i1p1f1_CMIP6', 'CESM2_2_CMIP6',],
#     'UKESM1-0-LL_5_CMIP6': [
#         'ACCESS-CM2_r1i1p1f1_CMIP6', 'ACCESS-ESM1-5_3_CMIP6', 'HadGEM3-GC31-LL_r1i1p1f3_CMIP6'],
# }

# same institution only
exclude_models = {
    'ACCESS-CM2_r1i1p1f1_CMIP6': [
        'ACCESS-ESM1-5_3_CMIP6'],
    'ACCESS-ESM1-5_3_CMIP6': [
        'ACCESS-CM2_r1i1p1f1_CMIP6'],
    'AWI-CM-1-1-MR_r1i1p1f1_CMIP6': [
        ],
    'BCC-CSM2-MR_r1i1p1f1_CMIP6': [
        ],
    'CAMS-CSM1-0_2_CMIP6': [
        ],
    'CESM2-WACCM_r1i1p1f1_CMIP6': [
        'CESM2_2_CMIP6'],
    'CESM2_2_CMIP6': [
        'CESM2-WACCM_r1i1p1f1_CMIP6'],
    'CNRM-CM6-1-HR_r1i1p1f2_CMIP6': [
        'CNRM-CM6-1_6_CMIP6', 'CNRM-ESM2-1_5_CMIP6'],
    'CNRM-CM6-1_6_CMIP6': [
        'CNRM-CM6-1-HR_r1i1p1f2_CMIP6', 'CNRM-ESM2-1_5_CMIP6'],
    'CNRM-ESM2-1_5_CMIP6': [
        'CNRM-CM6-1-HR_r1i1p1f2_CMIP6', 'CNRM-CM6-1_6_CMIP6'],
    'CanESM5-CanOE_3_CMIP6': [
        'CanESM5_50_CMIP6'],
    'CanESM5_50_CMIP6': [
        'CanESM5-CanOE_3_CMIP6'],
    'EC-Earth3-Veg_3_CMIP6': [
        'EC-Earth3_7_CMIP6'],
    'EC-Earth3_7_CMIP6': [
        'EC-Earth3-Veg_3_CMIP6'],
    'FGOALS-f3-L_r1i1p1f1_CMIP6': [
        'FGOALS-g3_r1i1p1f1_CMIP6'],
    'FGOALS-g3_r1i1p1f1_CMIP6': [
        'FGOALS-f3-L_r1i1p1f1_CMIP6'],
    'FIO-ESM-2-0_3_CMIP6': [],
    'GFDL-ESM4_r1i1p1f1_CMIP6': [],
    'GISS-E2-1-G_r1i1p3f1_CMIP6': [],
    'HadGEM3-GC31-LL_r1i1p1f3_CMIP6': [
        'UKESM1-0-LL_5_CMIP6'],
    'INM-CM4-8_r1i1p1f1_CMIP6': [
        'INM-CM5-0_r1i1p1f1_CMIP6'],
    'INM-CM5-0_r1i1p1f1_CMIP6': [
        'INM-CM4-8_r1i1p1f1_CMIP6'],
    'IPSL-CM6A-LR_6_CMIP6': [
        ],
    'KACE-1-0-G_r1i1p1f1_CMIP6': [],
    'MCM-UA-1-0_r1i1p1f2_CMIP6': [],
    'MIROC6_3_CMIP6': [
        'MIROC-ES2L_r1i1p1f2_CMIP6'],
    'MIROC-ES2L_r1i1p1f2_CMIP6': [
        'MIROC6_3_CMIP6'],
    'MPI-ESM1-2-HR_2_CMIP6': [
        'MPI-ESM1-2-LR_10_CMIP6'],
    'MPI-ESM1-2-LR_10_CMIP6': [
        'MPI-ESM1-2-HR_2_CMIP6'],
    'MRI-ESM2-0_r1i1p1f1_CMIP6': [],
    'NESM3_2_CMIP6': [
        ],
    'NorESM2-MM_r1i1p1f1_CMIP6': [
        ],
    'UKESM1-0-LL_5_CMIP6': [
        'HadGEM3-GC31-LL_r1i1p1f3_CMIP6'],
}

def crps_data(data, weights, models):
    """Calculate the CRPS.

    Parameters
    ----------
    data : np.array, shape (N,)
    weights : np.array, shape (N,)
        Exactly one element of data has to be NaN. This will be used to
        identify the index of the perfect model.

    Returns
    -------
    skill : float
    """
    assert np.isnan(weights).sum() == 1, 'exactly one weight has to be np.nan!'

    idx_perfect = np.where(np.isnan(weights))[0][0]

    model_perfect = models[idx_perfect]
    idx_del = [idx_perfect]
    # for model_del in exclude_models[model_perfect]:
    #     idx_del = np.concatenate([idx_del, np.where(models == model_del)[0]])
    idx_sel = np.delete(np.arange(data.shape[-1]), idx_del)

    data_ref = data[..., idx_perfect]
    data_test = data[..., idx_sel]
    weights_test = weights[idx_sel]

    baseline = crps_ensemble(data_ref, data_test)
    weighted = crps_ensemble(data_ref, data_test, weights=weights_test)
    return (baseline - weighted) / baseline


def crps_xarray(ds, varn):
    """
    Handle ensemble members and call CRPS calculation.

    Parameters
    ----------
    ds : xarray.Dataset
    varn : string

    Returns
    -------
    skill : xarray.DataArray
    """
    return xr.apply_ufunc(
        crps_data, ds[varn], ds['weights_mean'], ds['model'],
        input_core_dims=[['model'], ['model'], ['model']],
        vectorize=True,
    )


def read_data(filename, path):
    filename += '.nc' if not filename.endswith('.nc') else ''
    ds = xr.open_dataset(os.path.join(path, filename))
    ds = area_weighted_mean(ds)
    varn = ds.attrs['target'] + '_mean'
    return ds, varn


def main():
    """load files and call functions"""

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.95)
    colors = sns.color_palette('colorblind', 4)
    colors = np.array(colors)[np.array([0, 2, 1, 3])]

    ds_list = []
    trends = ['000', '033', '050', '066', '100']
    for idx, (fn126, fn585) in enumerate(zip(FILENAMES126, FILENAMES585)):
        xx = idx * 3
        ds, varn = read_data(fn126, LOADPATH)
        skill126 = crps_xarray(ds, varn)
        h1 = boxplot(ax, xx, data=skill126.data,
                     showdots=False, showmean=False,
                     color=colors[0],
                     # box_quantiles=(1/6., 5/6.)
        )

        ds, varn = read_data(fn585, LOADPATH)
        skill585 = crps_xarray(ds, varn)
        h4 = boxplot(ax, xx+1, data=skill585.data,
                     showdots=False, showmean=False,
                     color=colors[3],
                     # box_quantiles=(1/6., 5/6.)
        )

        skill126 = skill126.expand_dims({'ssp': ['ssp126']})
        skill585 = skill585.expand_dims({'ssp': ['ssp585']})
        ds = xr.concat([skill126, skill585], dim='ssp')
        ds = ds.expand_dims({'trend': [trends[idx]]})
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='trend').to_dataset(name='skill')

    ax.grid(axis='y')
    ax.axhline(0, color='k', zorder=2)

    ax.set_xticks(np.arange(len(FILENAMES126))*3+.5)
    # ax.set_xticklabels([
    #     '0\% trend', '33\% trend', '50\% trend',
    #     '66\% trend', '100\% trend'], rotation=30, ha='right')
    ax.set_xticklabels([
        '\n'.join([
            '\\textbf{0\% tasTREND}',
            '25\% tasANOM',
            '25\% tasSTD',
            '25\% pslANOM',
            '25\% pslSTD']),

        '\n'.join([
            '\\textbf{33\% tasTREND}',
            '17\% tasANOM',
            '17\% tasSTD',
            '17\% pslANOM',
            '17\% pslSTD']),

        '\n'.join([
            '\\textbf{50\% tasTREND}',
            '13\% tasANOM',
            '13\% tasSTD',
            '13\% pslANOM',
            '13\% pslSTD']),

        '\n'.join([
            '\\textbf{66\% tasTREND}',
            '8\% tasANOM',
            '8\% tasSTD',
            '8\% pslANOM',
            '8\% pslSTD']),

        '\n'.join([
            '\\textbf{100\% tasTREND}',
            '0\% tasANOM',
            '0\% tasSTD',
            '0\% pslANOM',
            '0\% pslSTD'])],
        multialignment='left')

    ax.set_ylim(-1, 1)
    ax.set_yticklabels(np.around(ax.get_yticks() * 100).astype(int))
    ax.set_ylabel('CRPSS (\%)', labelpad=-3)
    ax.set_title('\\textbf{(b) Performance weighting perfect model test skill 2081-2100}',
                 loc='left', fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join(SAVEPATH, SAVENAME), dpi=300)
    ds.to_netcdf(os.path.join(SAVEPATH, SAVENAME.replace('.png', '.nc')))
    plt.savefig(os.path.join(SAVEPATH2, SAVENAME.replace('.png', '.pdf')), dpi=300)


if __name__ == "__main__":
    main()
