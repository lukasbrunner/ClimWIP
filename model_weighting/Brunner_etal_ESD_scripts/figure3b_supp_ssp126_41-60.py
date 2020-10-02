#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-09-12 18:03:47 lukas>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Plot a map of the CRPS change between weighted and unweighted for a pseudo observation setting.
"""
import os
import warnings
import numpy as np
import xarray as xr
import matplotlib as mpl
import regionmask
mpl.use('Agg')
mpl.rc('text', usetex=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from glob import glob
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from properscoring import crps_ensemble

from model_weighting.core.utils import read_config
from model_weighting.core.utils_xarray import flip_antimeridian, area_weighted_mean

warnings.filterwarnings('ignore')

SAVEPATH = 'figures'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'
LOADPATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6_r1'
FILENAME = 'tas_global_pseudo_obs_ssp126_050_*-CMIP5-RCP26_41-60_ind.nc'


def get_skill(da, da_obs, weights):
    crps_baseline = xr.apply_ufunc(
        crps_ensemble, da_obs, da,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon', 'model_ensemble']],
        output_core_dims=[['lat', 'lon']],
    )

    crps_weighted = xr.apply_ufunc(
        crps_ensemble, da_obs, da,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon', 'model_ensemble']],
        output_core_dims=[['lat', 'lon']],
        kwargs={'weights': np.tile(weights, (da['lat'].size, da['lon'].size, 1))},
    )
    return (crps_baseline - crps_weighted) / crps_baseline * 100


def read_obs(ds, cfg):
    obs_id = cfg.obs_id
    obs_path = cfg.obs_path

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

    if cfg.target_mask == 'sea':
        sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 0
        ds_var = ds_var.where(sea_mask)
    elif cfg.target_mask == 'land':
        land_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 1
        ds_var = ds_var.where(land_mask)

    return ds_var.squeeze()


def print_global_mean(da):

    print(area_weighted_mean(da).data)

    sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(da, wrap_lon=180) == 0
    print('Land', area_weighted_mean(da.where(sea_mask)).data)

    land_mask = np.isnan(regionmask.defined_regions.natural_earth.land_110.mask(da, wrap_lon=180))
    print('Ocean', area_weighted_mean(da.where(land_mask)).data)


def plot_maps_mean(da, title=None, cmap='PuOr'):
    print_global_mean(da)

    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(9/1.2, 5.8/1.2))
    fig.subplots_adjust(left=.07, right=.97, bottom=0., top=.95)

    da.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': ''},
        cmap=cmap,
        robust=True,
        levels=11,
        vmax=25.,
        center=0.,
        extend='both'
    )

    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    longitude_formatter = LongitudeFormatter()
    latitude_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(longitude_formatter)
    ax.yaxis.set_major_formatter(latitude_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.coastlines()

    ax.set_title('\\textbf{(a) Combined weighting perfect model test median skill: SSP1-2.5, 2041-2060 (\%)}', loc='left',
                 fontdict={'size': 'large', 'weight': 'bold'})

    plt.savefig(os.path.join(SAVEPATH, 'figure3b_supp_ssp126_41-60.png'), dpi=300)
    plt.savefig(os.path.join(SAVEPATH2, 'figure3b_supp_ssp126_41-60.pdf'), dpi=300)


def main():
    skill_list = []
    for filename in glob(os.path.join(LOADPATH, FILENAME)):
        ds = xr.open_dataset(filename)

        varn = ds.attrs['target']
        if 'weights_mean' in ds:
            ds = ds.drop_dims(('model_ensemble', 'perfect_model_ensemble'))
            ds = ds.rename({'model': 'model_ensemble', 'perfect_model': 'perfect_model_ensemble',
                            f'{varn}_mean': varn, 'weights_mean': 'weights'})

        cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
        # log_parser(cfg)

        ds_obs = read_obs(ds, cfg)
        skill = get_skill(ds[varn], ds_obs, ds['weights'])
        skill_list.append(skill)

    skill = xr.concat(skill_list, dim='perfect_model_ensemble')

    plot_maps_mean(
        skill.median('perfect_model_ensemble'),
        f'Median of relative CRPS change (%)', cmap='PuOr_r')


if __name__ == '__main__':
    main()
