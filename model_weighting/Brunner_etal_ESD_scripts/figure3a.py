#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import regionmask
import numpy as np
import xarray as xr
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
import seaborn as sns
from glob import glob

from utils_python.xarray import area_weighted_mean, flip_antimeridian
from model_weighting.core.utils import read_config

from boxplot import boxplot

SAVEPATH = '/home/lukbrunn/Documents/Scripts/climWIP_clean_paper/model_weighting/scripts_paper/revision1/figures'
LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FILENAMES = [
    'tas_global_pseudo_obs_ssp126_050_*-CMIP5-RCP26_41-60_ind',
    'tas_global_pseudo_obs_ssp585_050_*-CMIP5-RCP85_41-60_ind',
    '',
    'tas_global_pseudo_obs_ssp126_050_*-CMIP5-RCP26_81-00_ind',
    'tas_global_pseudo_obs_ssp585_050_*-CMIP5-RCP85_81-00_ind',
]
SAVENAME = 'figure3a.png'

# wether to calculate the CRPS on the full 2D field and only then take the average
MEAN_OF_CRPS = False


def get_skill(da, da_obs, weights, mean_of_crps):
    if mean_of_crps:
        crps_baseline = area_weighted_mean(xr.apply_ufunc(
            crps_ensemble, da_obs, da,
            input_core_dims=[['lat', 'lon'], ['lat', 'lon', 'model_ensemble']],
            output_core_dims=[['lat', 'lon']],
        ))

        crps_weighted = area_weighted_mean(xr.apply_ufunc(
            crps_ensemble, da_obs, da,
            input_core_dims=[['lat', 'lon'], ['lat', 'lon', 'model_ensemble']],
            output_core_dims=[['lat', 'lon']],
            kwargs={'weights': np.tile(weights, (da['lat'].size, da['lon'].size, 1))},
        ))
    else:
        crps_baseline = xr.apply_ufunc(
            crps_ensemble,
            area_weighted_mean(da_obs),
            area_weighted_mean(da),
            input_core_dims=[[], ['model_ensemble']],
        )

        crps_weighted = xr.apply_ufunc(
            crps_ensemble,
            area_weighted_mean(da_obs),
            area_weighted_mean(da),
            input_core_dims=[[], ['model_ensemble']],
            kwargs={'weights': weights},
        )

    return (crps_baseline - crps_weighted) / crps_baseline


def read_obs(ds, cfg):

    if cfg.obs_id is None:
        return None, None

    if isinstance(cfg.obs_id, str):
        cfg.obs_id = [cfg.obs_id]
        cfg.obs_path = [cfg.obs_path]

    ds_list = []
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):

        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(cfg.target_diagnostic, obs_id))
        ds_var = xr.open_dataset(filename, use_cftime=True)[cfg.target_diagnostic].load()

        if cfg.target_season != 'ANN':
            ds_var = ds_var.isel(time=ds_var['time.season'] == cfg.target_season)

        ds_var = (ds_var.sel(time=slice(str(cfg.target_startyear), str(cfg.target_endyear))).mean('time') -
                  ds_var.sel(time=slice(str(cfg.target_startyear_ref), str(cfg.target_endyear_ref))).mean('time'))

        ds_var = flip_antimeridian(ds_var)
        ds_var = ds_var.sel(lon=ds['lon'], lat=ds['lat'])

        if ds_var.max() > 100:  # assume Kelvin
            ds_var -= 273.15
            ds_var.attrs['unit'] = 'degC'

        ds_list.append(ds_var)

    ds_var = xr.concat(ds_list, dim='dataset')

    if cfg.target_mask == 'sea':
        sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 0
        ds_var = ds_var.where(sea_mask)
    elif cfg.target_mask == 'land':
        land_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 1
        ds_var = ds_var.where(land_mask)

    return ds_var.squeeze()


def main():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.09, right=.99, bottom=.09, top=.94)
    c1, c2, _, c3 = sns.color_palette('colorblind', 4)
    colors = [c1, c3, ''] * 2
    alphas = [1, 1, ''] * 2

    #  skills = []
    for xx, filename_pattern in enumerate(FILENAMES):
        if filename_pattern == '':
            continue
        filename_pattern = os.path.join(LOADPATH, filename_pattern)
        if not filename_pattern.endswith('.nc'):
            filename_pattern += '.nc'
        filenames = glob(filename_pattern)
        skill_list = []
        assert len(filenames) > 0, filename_pattern
        for filename in filenames:
            ds = xr.open_dataset(filename)
            varn = ds.attrs['target']
            if 'weights_mean' in ds:
                ds = ds.drop_dims(('model_ensemble', 'perfect_model_ensemble'))
                ds = ds.rename({'model': 'model_ensemble', 'perfect_model': 'perfect_model_ensemble',
                                f'{varn}_mean': varn, 'weights_mean': 'weights'})
            cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
            # log_parser(cfg)
            ds_obs = read_obs(ds, cfg)
            skill = get_skill(ds[varn], ds_obs, ds['weights'], MEAN_OF_CRPS)
            skill_list.append(skill)

        skill = xr.concat(skill_list, dim='perfect_model_ensemble')
        boxplot(ax, xx, data=skill.data, showmean=False, showdots=False, color=colors[xx], alpha=alphas[xx])

        for fn, ss in zip(filenames, skill.data):
            pm = fn.split('_')[7]
            print(f'"{pm}": {ss},')

        # idxs = np.argsort(skill.data)[::-1]
        # print('---')
        # for idx in idxs[:]:
        #     if skill.data[idx] < 0:
        #         print(f'{os.path.basename(filenames[idx])}: {skill.data[idx]:.1%}')
        # # for idx in idxs[-3:]:
        # #     print(f'{os.path.basename(filenames[idx])}: {skill.data[idx]:.1%}')

        # print(skill.median().data)
    ax.grid(axis='y')
    ax.axhline(0, color='k', zorder=2)

    ax.set_xticks([.5, 3.5])
    ax.set_xticklabels(['2041-2060', '2081-2100'])

    h1 = boxplot(ax, [], data=skill.data, return_handle=True, showmean=False, showdots=False, color=c1, alpha=1)
    h3 = boxplot(ax, [], data=skill.data, return_handle=True, showmean=False, showdots=False, color=c3, alpha=1)
    leg = plt.legend([h1, h3], ['SSP1-2.6', 'SSP5-8.5'], loc='lower left', title='Median, 50\%, 90\%')
    leg._legend_box.align = 'left'

    ax.set_ylim(-1, 1)
    ax.set_yticklabels(np.around(ax.get_yticks() * 100).astype(int))
    ax.set_ylabel('CRPSS (\%)', labelpad=-3)

    ax.set_title('\\textbf{(a) Combined weighting perfect model test skill (CMIP5)}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    ax.annotate('Shown as map in (b)', (1, .22), (1.3, .65), arrowprops={'arrowstyle': '->'}, zorder=999)

    plt.savefig(os.path.join(SAVEPATH, SAVENAME), dpi=300)
    SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
    plt.savefig(os.path.join(SAVEPATH2, SAVENAME.replace('.png', '.pdf')), dpi=300)


if __name__ == '__main__':
    main()
