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
trend = '050'

SUBSET_CMIP5 = [
'bcc-csm1-1-m_r1i1p1_CMIP5', 'bcc-csm1-1_r1i1p1_CMIP5', 'BNU-ESM_r1i1p1_CMIP5', 'CanESM2_r1i1p1_CMIP5', 'CCSM4_r1i1p1_CMIP5', 'CESM1-CAM5_r1i1p1_CMIP5', 'CNRM-CM5_r1i1p1_CMIP5', 'CSIRO-Mk3-6-0_r1i1p1_CMIP5', 'EC-EARTH_r8i1p1_CMIP5', 'FGOALS-g2_r1i1p1_CMIP5', 'GFDL-CM3_r1i1p1_CMIP5', 'GFDL-ESM2G_r1i1p1_CMIP5', 'GFDL-ESM2M_r1i1p1_CMIP5', 'GISS-E2-H_r1i1p1_CMIP5', 'GISS-E2-R_r1i1p1_CMIP5', 'HadGEM2-AO_r1i1p1_CMIP5', 'HadGEM2-ES_r1i1p1_CMIP5', 'IPSL-CM5A-LR_r1i1p1_CMIP5', 'IPSL-CM5A-MR_r1i1p1_CMIP5', 'MIROC5_r1i1p1_CMIP5', 'MIROC-ESM-CHEM_r1i1p1_CMIP5', 'MIROC-ESM_r1i1p1_CMIP5', 'MPI-ESM-LR_r1i1p1_CMIP5', 'MPI-ESM-MR_r1i1p1_CMIP5', 'MRI-CGCM3_r1i1p1_CMIP5', 'NorESM1-M_r1i1p1_CMIP5', 'NorESM1-ME_r1i1p1_CMIP5'
]

ansesctry = {
    'EC-EARTH': ['EC-Earth3-Veg', 'EC-Earth3'],
    'bcc-csm1-1-m': ['BCC-CSM2-MR'],
    'bcc-csm1-1': ['BCC-CSM2-MR'],
    'BNU-ESM': [],
    'CanESM2': ['CanESM5', 'CanESM5-CanOE'],
    'CCSM4': [],
    'CESM1-CAM5': ['CESM2-WACCM', 'CESM2'],
    'CNRM-CM5': ['CNRM-CM6-1-HR', 'CNRM-CM6-1', 'CNRM-ESM2-1'],
    'CSIRO-Mk3-6-0': [],
    'FGOALS-g2': ['FGOALS-f3-L', 'FGOALS-g3'],
    'GFDL-CM3': ['GFDL-ESM4'],
    'GFDL-ESM2G': ['GFDL-ESM4'],
    'GFDL-ESM2M': ['GFDL-ESM4'],
    'GISS-E2-H': ['GISS-E2-1-G'],
    'GISS-E2-R': ['GISS-E2-1-G'],
    'HadGEM2-AO': ['HadGEM3-GC31-LL', 'UKESM1-0-LL'],
    'HadGEM2-ES': ['HadGEM3-GC31-LL', 'UKESM1-0-LL'],
    'IPSL-CM5A-LR': ['IPSL-CM6A-LR'],
    'IPSL-CM5A-MR': ['IPSL-CM6A-LR'],
    'MIROC5': ['MIROC-ES2L', 'MIROC6'],
    'MIROC-ESM-CHEM': ['MIROC-ES2L', 'MIROC6'],
    'MIROC-ESM': ['MIROC-ES2L', 'MIROC6'],
    'MPI-ESM-LR': ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR'],
    'MPI-ESM-MR': ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR'],
    'MRI-CGCM3': ['MRI-ESM2-0'],
    'NorESM1-M': ['NorESM2-MM'],
    'NorESM1-ME': ['NorESM2-MM'],
}

SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'


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


def load_data(*args, filenames=None):
    cfg = create_pseudo_cfg(*args)
    if filenames is None:
        filenames = get_filenames(cfg)['tas']
    else:
        filenames = {key: str(filenames.sel(model_ensemble=key).data) for key in filenames['model_ensemble'].data}

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
            ds = ds.groupby('time.year').mean('time')
            ds = ds.sel(year=slice('1950', '2100'))
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


def load_weights(ssp):
    fn = f'data_temp/tas_global_pseudo_obs_{ssp}_{trend}_81-00_merged.nc'
    return xr.open_dataset(fn)['weights']


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



def plot_skill(model):

    ssp126_4160 = {
        "EC-EARTH": 0.13509111442915908,
        "bcc-csm1-1-m": 0.17305705898935395,
        "bcc-csm1-1": 0.13933601295961723,
        "BNU-ESM": 0.15126041715595262,
        "CanESM2": 0.15576509929024357,
        "CCSM4": 0.030013767028635153,
        "CESM1-CAM5": 0.15095239148832845,
        "CNRM-CM5": 0.17100627797158074,
        "CSIRO-Mk3-6-0": 0.1376933885984476,
        "FGOALS-g2": 0.18215754823850264,
        "GFDL-CM3": 0.021816460970221774,
        "GFDL-ESM2G": 0.07691281544649466,
        "GFDL-ESM2M": 0.08622900955147449,
        "GISS-E2-H": 0.2594083642818405,
        "GISS-E2-R": 0.1632444740859207,
        "HadGEM2-AO": -0.031192983817657054,
        "HadGEM2-ES": 0.13144781684232082,
        "IPSL-CM5A-LR": 0.3170087843706667,
        "IPSL-CM5A-MR": 0.16531169770491938,
        "MIROC5": 0.09524089500076785,
        "MIROC-ESM-CHEM": -0.05721678610758046,
        "MIROC-ESM": -0.300140307348579,
        "MPI-ESM-LR": 0.3184101281757854,
        "MPI-ESM-MR": 0.06954396156334193,
        "MRI-CGCM3": 0.3494653979787956,
        "NorESM1-M": 0.1342342710274076,
        "NorESM1-ME": 0.21182703606168735}

    ssp585_4160 = {
        "EC-EARTH": 0.21537579376200688,
        "bcc-csm1-1-m": 0.1731942537739601,
        "bcc-csm1-1": 0.1938346717099523,
        "BNU-ESM": 0.041354277264961666,
        "CanESM2": 0.3024068963927599,
        "CCSM4": 0.25858069702636843,
        "CESM1-CAM5": -0.1783937411528149,
        "CNRM-CM5": 0.1407554015196773,
        "CSIRO-Mk3-6-0": 0.2712229831005372,
        "FGOALS-g2": 0.3085665109238221,
        "GFDL-CM3": 0.13453932540026792,
        "GFDL-ESM2G": 0.10190011182516626,
        "GFDL-ESM2M": 0.25914277797297686,
        "GISS-E2-H": 0.47019899891690903,
        "GISS-E2-R": 0.31220634652054247,
        "HadGEM2-AO": -0.07014210719866594,
        "HadGEM2-ES": 0.2321015812538282,
        "IPSL-CM5A-LR": -0.1632891044024148,
        "IPSL-CM5A-MR": 0.0317708749509122,
        "MIROC5": 0.0819163185975342,
        "MIROC-ESM-CHEM": -0.16930748528615377,
        "MIROC-ESM": -0.34839851819195516,
        "MPI-ESM-LR": 0.4998355386914368,
        "MPI-ESM-MR": 0.26498404167038997,
        "MRI-CGCM3": 0.4103424465694992,
        "NorESM1-M": 0.22670289557866982,
        "NorESM1-ME": 0.28067928268372344}

    ssp126_8100 = {
        "EC-EARTH": 0.13155818793136909,
        "bcc-csm1-1-m": 0.16328131303696566,
        "bcc-csm1-1": 0.16568004817600288,
        "BNU-ESM": 0.18797104010620638,
        "CanESM2": 0.07178144746044778,
        "CCSM4": 0.04669828235065317,
        "CESM1-CAM5": -0.08499661300787516,
        "CNRM-CM5": 0.23032027610757988,
        "CSIRO-Mk3-6-0": -0.22282706850660716,
        "FGOALS-g2": 0.12193191281682948,
        "GFDL-CM3": 0.023724192290311964,
        "GFDL-ESM2G": 0.07445130087870865,
        "GFDL-ESM2M": 0.13325427575528198,
        "GISS-E2-H": 0.24231509616500638,
        "GISS-E2-R": 0.14137000246922043,
        "HadGEM2-AO": -0.11751713205248049,
        "HadGEM2-ES": -0.14709295462061212,
        "IPSL-CM5A-LR": 0.19975092234045416,
        "IPSL-CM5A-MR": 0.05444015744268936,
        "MIROC5": 0.09228993755510796,
        "MIROC-ESM-CHEM": -0.19911604004171135,
        "MIROC-ESM": -0.4369180812729563,
        "MPI-ESM-LR": 0.24292723327735713,
        "MPI-ESM-MR": 0.09026278693087744,
        "MRI-CGCM3": 0.4781211806519122,
        "NorESM1-M": 0.23069453189264832,
        "NorESM1-ME": 0.3145576599031033}

    ssp585_8100 = {
        "EC-EARTH": -0.0093454392370977,
        "bcc-csm1-1-m": 0.15261894797658845,
        "bcc-csm1-1": 0.22847629435176994,
        "BNU-ESM": 0.08553658989517429,
        "CanESM2": 0.11697934975735849,
        "CCSM4": 0.12822824489822265,
        "CESM1-CAM5": 0.19608610361803874,
        "CNRM-CM5": 0.19383680080945856,
        "CSIRO-Mk3-6-0": 0.1884326453202373,
        "FGOALS-g2": 0.3008296629004765,
        "GFDL-CM3": 0.1491413891777341,
        "GFDL-ESM2G": 0.17707130452357728,
        "GFDL-ESM2M": 0.2349813058327707,
        "GISS-E2-H": 0.3349706056137517,
        "GISS-E2-R": 0.21486345820581057,
        "HadGEM2-AO": 0.13646009545672083,
        "HadGEM2-ES": 0.20681342091312066,
        "IPSL-CM5A-LR": -0.10567165583214941,
        "IPSL-CM5A-MR": 0.07018073966528666,
        "MIROC5": 0.12715244266575934,
        "MIROC-ESM-CHEM": -0.11144900490485932,
        "MIROC-ESM": -0.3917994230696753,
        "MPI-ESM-LR": 0.5615343842807429,
        "MPI-ESM-MR": 0.31885823283233844,
        "MRI-CGCM3": 0.49419563297816943,
        "NorESM1-M": 0.2989561209735816,
        "NorESM1-ME": 0.3560864092595498}

    key = [mm for mm in ssp126_4160 if model == mm]
    assert len(key) == 1
    key = key[0]

    return ssp585_4160[key], ssp585_8100[key], ssp126_4160[key], ssp126_8100[key]


def plot_single(ax, cmip5_rcp26, cmip6_ssp126, cmip5_rcp85, cmip6_ssp585,
                weights_ssp126=None, weights_ssp585=None, xticks=False):

    colors = sns.color_palette('colorblind', 4)
    xx = cmip5_rcp26['year']
    xx2 = cmip5_rcp26['year'].sel(year=slice('2015', None))

    # --- RCP26/SSP126 ---
    plot_line(
        ax, xx,
        cmip5_rcp26.data,
        colors[0], ls='--')

    plot_shading(
        ax, xx,
        quantile(cmip6_ssp126.data.swapaxes(0, 1), 1/6.),
        quantile(cmip6_ssp126.data.swapaxes(0, 1), 5/6.), sns.xkcd_rgb['greyish'])
    plot_line(
        ax, xx,
        cmip6_ssp126.mean('model').data,
        # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5),
        sns.xkcd_rgb['greyish'])

    plot_shading(
        ax, xx2,
        quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
                 weights=weights_ssp126),
        quantile(cmip6_ssp126.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
                 weights=weights_ssp126), colors[0])
    plot_line(
        ax, xx2,
        np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp126.sel(year=slice('2015', None)).data),
                                   axis=0, weights=weights_ssp126)),
        # quantile(cmip6_ssp126.data.swapaxes(0, 1), .5, weights=weights_ssp126),
        colors[0])

    # --- RCP85/SSP585
    plot_line(
        ax, xx,
        cmip5_rcp85.data,
        colors[3], ls='--')

    if weights_ssp585 is not None:
        plot_shading(
            ax, xx,
            quantile(cmip6_ssp585.data.swapaxes(0, 1), 1/6.),
            quantile(cmip6_ssp585.data.swapaxes(0, 1), 5/6.), sns.xkcd_rgb['greyish'])
        plot_line(
            ax, xx,
            cmip6_ssp585.mean('model').data,
            # quantile(cmip6_ssp585.data.swapaxes(0, 1), .5),
            sns.xkcd_rgb['greyish'])

    plot_shading(
        ax, xx2,
        quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 1/6.,
                 weights=weights_ssp585),
        quantile(cmip6_ssp585.sel(year=slice('2015', None)).data.swapaxes(0, 1), 5/6.,
                 weights=weights_ssp585), colors[3])
    plot_line(
        ax, xx2,
        np.ma.filled(np.ma.average(np.ma.masked_invalid(cmip6_ssp585.sel(year=slice('2015', None)).data),
                                   axis=0, weights=weights_ssp585)),
        # quantile(cmip6_ssp585.data.swapaxes(0, 1), .5, weights=weights_ssp585),
        colors[3])

    ax.set_xlim(1975, 2100)
    ax.set_ylim(-1, 5)
    ax.grid(zorder=0, axis='y')

    ax.set_yticks([0, 2, 4])
    ax.set_xticks([1975, 2025, 2075, 2100])


def main():

    filenames_ssp126 = xr.open_dataset(
        os.path.join(
            '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1',
            'tas_global_{ssp}_050_81-00.nc'.format(
                ssp='ssp126')))['filename']
    filenames_ssp585 = xr.open_dataset(
        os.path.join(
            '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1',
            'tas_global_{ssp}_050_81-00.nc'.format(
                ssp='ssp585')))['filename']

    cmip5_rcp26 = load_data('CMIP5', 'rcp26', SUBSET_CMIP5, '/net/atmos/data/cmip5-ng')
    cmip6_ssp126 = load_data('CMIP6', 'ssp126', None, '/net/ch4/data/cmip6-Next_Generation',
                             filenames=filenames_ssp126)

    cmip5_rcp85 = load_data('CMIP5', 'rcp85', SUBSET_CMIP5, '/net/atmos/data/cmip5-ng')
    cmip6_ssp585 = load_data('CMIP6', 'ssp585', None, '/net/ch4/data/cmip6-Next_Generation',
                             filenames=filenames_ssp585)

    weights_ssp126 = load_weights('ssp126')
    models = np.intersect1d(weights_ssp126['model'].data, cmip6_ssp126['model'].data)
    weights_ssp126 = weights_ssp126.sel(model=models)
    cmip6_ssp126 = cmip6_ssp126.sel(model=models)

    assert len(models) == 33

    weights_ssp585 = load_weights('ssp585')
    models = np.intersect1d(weights_ssp585['model'].data, cmip6_ssp585['model'].data)
    weights_ssp585 = weights_ssp585.sel(model=models)
    cmip6_ssp585 = cmip6_ssp585.sel(model=models)

    assert len(models) == 33

    fig, axs = plt.subplots(7, 4, sharex=True, sharey=True, figsize=(10, 13),
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.subplots_adjust(left=.07, right=.99, bottom=.05, top=.95)

    for pseudo_obs, ax in zip(natsorted(weights_ssp126['pseudo_obs'].data, alg=ns.IC), axs.ravel()):

        # remove direct ansestors
        exclude = ansesctry[pseudo_obs]
        model_ensemble = [mm for mm in cmip6_ssp126['model'].copy().data]
        models = [mm.split('_')[0] for mm in model_ensemble]
        for ee in exclude:
            idx = models.index(ee)
            assert np.isnan(weights_ssp126.sel(pseudo_obs=pseudo_obs, model=model_ensemble[idx]).data)
            models.pop(idx)
            model_ensemble.pop(idx)

        assert len(model_ensemble) ==  33 - len(exclude)

        plot_single(
            ax,
            cmip5_rcp26.sel(model=pseudo_obs), cmip6_ssp126.sel(model=model_ensemble),
            cmip5_rcp85.sel(model=pseudo_obs), cmip6_ssp585.sel(model=model_ensemble),
            weights_ssp126=weights_ssp126.sel(pseudo_obs=pseudo_obs, model=model_ensemble),
            weights_ssp585=weights_ssp585.sel(pseudo_obs=pseudo_obs, model=model_ensemble),
            xticks=False)

        ax.set_xlim(1975, 2099)
        ax.set_xticks([1980, 2020, 2060])
        ax.set_ylim(-1, 5)
        ax.grid(zorder=0, axis='y')
        ax.text(1977, 4.2, f'\\textbf{{{pseudo_obs}}}')

        values = plot_skill(pseudo_obs)
        ax.text(1977, 3.8, '{:.1f}\%; {:.1f}\% \n{:.1f}\%; {:.1f}\%'.format(*[100*vv for vv in values]), va='top')

    axs[3, 0].set_ylabel('Temperature change (°C) relative to 1995-2014', fontsize='x-large', labelpad=10)
    fig.suptitle('\\textbf{Combined weighting based on pseudo-observations from CMIP5}', fontsize='xx-large')

    plt.savefig(f'figures/figure2_supp_all_{trend}.png', dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, f'figure2_supp_all_{trend}.pdf'), dpi=300)
    plt.close()

    # for pseudo_obs, ax in zip(weights_ssp126['pseudo_obs'].data, axs.ravel()):
    #     fig, ax = plt.subplots()

    #     plot_single(
    #         ax,
    #         cmip5_rcp26.sel(model=pseudo_obs), cmip6_ssp126,
    #         cmip5_rcp85.sel(model=pseudo_obs), cmip6_ssp585,
    #         weights_ssp126=weights_ssp126.sel(pseudo_obs=pseudo_obs),
    #         weights_ssp585=weights_ssp585.sel(pseudo_obs=pseudo_obs),
    #         xticks=False)

    #     ax.hlines([1.2], [1980], [2014], color='k', lw=2)
    #     ax.text(1996.5, 1.3, 'Diagnostic period', va='bottom', ha='center')
    #     ww = weights_ssp585.sel(pseudo_obs=pseudo_obs)
    #     idxs = np.argsort(ww.data)[::-1]
    #     lines = []
    #     for idx in idxs[:3]:
    #         line = f'{ww["model"].data[idx]}: {ww.data[idx]:.3f}'
    #         lines.append(line)
    #     lines.append('...')
    #     for idx in idxs[-3:]:
    #         line = f'{ww["model"].data[idx]}: {ww.data[idx]:.3f}'
    #         lines.append(line)
    #     ax.text(1978, 4.8, '\n'.join(lines), va='top', ha='left')
    #     ax.set_ylabel('Temperature change (K)')
    #     plt.savefig(f'figures/tests/timeseries_pseudo-obs_{pseudo_obs}_{trend}.png', dpi=300)
    #     plt.savefig(f'figures/tests/timeseries_pseudo-obs_{pseudo_obs}_{trend}.pdf', dpi=300)
    #     print(f'Saved. figures/tests/timeseries_pseudo-obs_{pseudo_obs}_{trend}.png')
    #     plt.close()


if __name__ == '__main__':
    main()
