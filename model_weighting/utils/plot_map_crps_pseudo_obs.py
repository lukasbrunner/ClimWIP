#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-01-29 14:33:45 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Plot a map of the CRPS change between weighted and unweighted for a pseudo observation setting.
"""
import os
import argparse
import warnings
import numpy as np
import xarray as xr
import matplotlib as mpl
import regionmask
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from glob import glob
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from properscoring import crps_ensemble

from model_weighting.core.utils import read_config, log_parser
from model_weighting.core.utils_xarray import quantile, flip_antimeridian, area_weighted_mean

warnings.filterwarnings('ignore')

period_ref = None  # slice('1979', '2014')

PLOTPATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../plots/maps_crps_pseudo_obs')
os.makedirs(PLOTPATH, exist_ok=True)
os.makedirs(os.path.join(PLOTPATH, 'single'), exist_ok=True)


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filenames', type=str, nargs='+',
        help='Valid weights file (should end with .nc)')
    parser.add_argument(
        '--plot-type', '-t', dest='ext', default='png', type=str,
        help=' '.join([
            'A valid plot extension specifiying the file type. A special case',
            'is "show" which will call plt.show() instead of saving']))
    args = parser.parse_args()

    if len(args.filenames) == 1:
        args.filenames = glob(args.filenames[0])

    return args


def read_obs(ds, cfg):
    if isinstance(cfg.obs_id, str):
        cfg.obs_id = [cfg.obs_id]
        cfg.obs_path = [cfg.obs_path]

    ds_list = []
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
        filename = os.path.join(obs_path, '{}_mon_{}_g025.nc'.format(cfg.target_diagnostic, obs_id))
        ds_var = xr.open_dataset(filename, use_cftime=True)[cfg.target_diagnostic].load()

        try:
            filename = os.path.join(obs_path, '{}_mon_{}_g025_future.nc'.format(cfg.target_diagnostic, obs_id))
            ds_var2 = xr.open_dataset(filename, use_cftime=True)[cfg.target_diagnostic].load()
            ds_var = xr.concat([ds_var, ds_var2], dim='time')
        except FileNotFoundError:
            return None

        ds_var = ds_var.drop_vars('height', errors='ignore')

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
        sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 1
        ds_var = ds_var.where(sea_mask)

    return ds_var.squeeze()


def weighted_mean(data, weights):
    return np.average(data, weights=weights)


def weighted_quantile(data, weights, q):
    if np.any(np.isnan(data)):
        return np.nan
    return quantile(data, q, weights=weights)


def plot_maps(da, da_obs, weights, cfg, fn):
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, axes = plt.subplots(ncols=3, nrows=5, subplot_kw={'projection': proj}, figsize=(10, 16))
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.93, hspace=.17, wspace=.1)

    diff = da - da_obs

    diff_best = diff.isel(model_ensemble=weights.argmax())
    diff_worst = diff.isel(model_ensemble=weights.argmin())
    model_ensemble = da['model_ensemble'].data

    vmax = np.max([
        np.abs(diff.quantile(.02, ('model_ensemble', 'lat', 'lon'))),
        np.abs(diff.quantile(.98, ('model_ensemble', 'lat', 'lon')))])

    wdiff_mean = xr.apply_ufunc(
        weighted_mean, diff, weights,
        input_core_dims=[['model_ensemble'], ['model_ensemble']],
        vectorize=True)

    wdiff_p10 = xr.apply_ufunc(
        weighted_quantile, diff, weights, .1,
        input_core_dims=[['model_ensemble'], ['model_ensemble'], []],
        vectorize=True)

    wdiff_p90 = xr.apply_ufunc(
        weighted_quantile, diff, weights, .9,
        input_core_dims=[['model_ensemble'], ['model_ensemble'], []],
        vectorize=True)

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

    # CRPS is the relative change in the CRPS
    crps = (crps_baseline - crps_weighted) / crps_baseline * 100

    # Bias is the relative change in the differences between absolute weighted and unweighted model-obs distances
    bias = (np.abs(wdiff_mean) - np.abs(diff.mean('model_ensemble'))) / np.abs(diff.mean('model_ensemble')) * 100

    rmse = np.sqrt((diff**2).mean('model_ensemble'))
    wrmse = np.sqrt(xr.apply_ufunc(
        weighted_mean, diff**2, weights,
        input_core_dims=[['model_ensemble'], ['model_ensemble']],
        vectorize=True))

    # The RMSE change is the relative change in the RMSE
    rmse_change = (wrmse - rmse) / rmse * 100

    da_obs.plot.pcolormesh(
        ax=axes[0, 0], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        cmap='Reds',
    )
    axes[0, 0].set_title('"True" temperature change')

    diff_best.plot.pcolormesh(
        ax=axes[0, 1], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    nn = model_ensemble[weights.argmax()]
    axes[0, 1].set_title(f'Difference best model\n{nn}')

    diff_worst.plot.pcolormesh(
        ax=axes[0, 2], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    nn = model_ensemble[weights.argmin()]
    axes[0, 2].set_title(f'Difference worst model\n{nn}')

    diff.mean('model_ensemble').plot.pcolormesh(
        ax=axes[1, 0], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        # cbar='RdBu_r',
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[1, 0].set_title('Differences mean')

    wdiff_mean.plot.pcolormesh(
        ax=axes[1, 1], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        # cbar='RdBu_r',
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[1, 1].set_title('Differences weighted mean')

    (wdiff_mean - diff.mean('model_ensemble')).plot.pcolormesh(
        ax=axes[1, 2], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax*.25,
        extend='both',
    )
    axes[1, 2].set_title('Change in the differences')

    diff.quantile(.1, 'model_ensemble').plot.pcolormesh(
        ax=axes[2, 0], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[2, 0].set_title('Differences 10th perc.')

    wdiff_p10.plot.pcolormesh(
        ax=axes[2, 1], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[2, 1].set_title('Differences weighted 10th perc.')

    (wdiff_p10 - diff.quantile(.1, 'model_ensemble')).plot.pcolormesh(
        ax=axes[2, 2], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax*.25,
        extend='both',
    )
    axes[2, 2].set_title('Change in the differences')

    diff.quantile(.9, 'model_ensemble').plot.pcolormesh(
        ax=axes[3, 0], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[3, 0].set_title('Differences 90th perc.')

    wdiff_p90.plot.pcolormesh(
        ax=axes[3, 1], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax,
        extend='both',
    )
    axes[3, 1].set_title('Differences weighted 90th perc.')

    (wdiff_p90 - diff.quantile(.9, 'model_ensemble')).plot.pcolormesh(
        ax=axes[3, 2], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': 'K'},
        center=0.,
        vmax=vmax*.25,
        extend='both',
    )
    axes[3, 2].set_title('Change in the differences')

    crps.plot.pcolormesh(
        ax=axes[4, 0], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': '%'},
        cmap='PuOr_r',
        # robust=True,
        vmax=50.,
        center=0.,
        extend='both',
    )
    axes[4, 0].set_title(f'Relative CRPS change ({area_weighted_mean(crps).data:.1f}%)')

    # bias.plot.pcolormesh(
    #     ax=axes[4, 1], transform=ccrs.PlateCarree(),
    #     cbar_kwargs={'orientation': 'horizontal',
    #                  'pad': .1,
    #                  'label': '%'},
    #     cmap='PuOr',
    #     # vmax=np.min([np.max(np.abs(bias.quantile((.02, .98)))), 50]),
    #     vmax=50.,
    #     center=0.,
    #     extend='both',
    # )
    # axes[4, 1].set_title(f'Relative change in mean bias ({area_weighted_mean(bias).data:.1f})')

    rmse_change.plot.pcolormesh(
        ax=axes[4, 1], transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': '%'},
        cmap='PuOr',
        # vmax=np.min([np.max(np.abs(rmse_change.quantile((.02, .98)))), 50]),
        vmax=50.,
        center=0.,
        extend='both',
    )
    axes[4, 1].set_title(f'Relative change in RMSE ({area_weighted_mean(rmse_change).data:.1f}%)')

    longitude_formatter = LongitudeFormatter()
    latitude_formatter = LatitudeFormatter()
    for ax in axes.ravel():
        ax.coastlines()
        if cfg.target_region == 'GLOBAL':
            ax.set_xticks(np.arange(-180, 181, 60))
            ax.set_yticks(np.arange(-90, 91, 30))
        else:
            ax.set_xticks(np.arange(np.floor(da['lon'].min()), da['lon'].max(), 10), crs=proj)
            ax.set_yticks(np.arange(np.floor(da['lat'].min()), da['lat'].max(), 10), crs=proj)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_formatter(longitude_formatter)
        ax.yaxis.set_major_formatter(latitude_formatter)
        ax.set_xlabel('')
        ax.set_ylabel('')

    title = ' '.join([
        f'Perfect model test: using {" ".join(cfg.obs_id)} (CMIP6) to weight the\n',
        f'{cfg.model_id} MME for {cfg.target_region} {cfg.target_diagnostic}',
        f'({cfg.target_startyear}-{cfg.target_endyear} minus',
        f'{cfg.target_startyear_ref}-{cfg.target_endyear_ref})'])
    fig.suptitle(title)

    plt.savefig(os.path.join(PLOTPATH, 'single', fn), dpi=300)
    plt.close()

    crps.name = 'CRPS'
    skill = crps.to_dataset()
    skill['Bias'] = bias
    skill['RMSE'] = rmse_change
    skill['perfect_model_ensemble'] = xr.DataArray(cfg.obs_id, dims='perfect_model_ensemble')

    return skill


def plot_maps_mean(da, global_, fn=None, title=None, cmap='PuOr'):
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(subplot_kw={'projection': proj})
    # fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.93, hspace=.17, wspace=.1)

    da.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': '%'},
        cmap=cmap,
        robust=True,
        levels=11,
        vmax=20.,
        center=0.,
        extend='both'
    )

    if global_:
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
    else:
        ax.set_xticks(np.arange(np.floor(da['lon'].min()), da['lon'].max(), 10), crs=proj)
        ax.set_yticks(np.arange(np.floor(da['lat'].min()), da['lat'].max(), 10), crs=proj)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    longitude_formatter = LongitudeFormatter()
    latitude_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(longitude_formatter)
    ax.yaxis.set_major_formatter(latitude_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.coastlines()

    if title is not None:
        title = f'{title} (Global mean: {area_weighted_mean(da).data:.1f}%)'
        ax.set_title(title)

    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join(PLOTPATH, fn), dpi=300)


def main():
    args = read_input()
    skill_list = []
    # ds_list = []
    for filename in args.filenames:
        ds = xr.open_dataset(filename)

        varn = ds.attrs['target']
        region = ds.attrs['region']

        cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
        log_parser(cfg)

        ds_obs = read_obs(ds, cfg)

        if ds_obs is None:
            continue

        fn = f'Mapplots_skill_{ds.attrs["config"]}.png'

        skill = plot_maps(ds[varn], ds_obs, ds['weights'], cfg, fn)
        # ds['model_ensemble'] = xr.DataArray(cfg.obs_id, dims='model_ensemble')

        skill_list.append(skill)
        # ds_list.appendd(ds)

    skill = xr.concat(skill_list, dim='perfect_model_ensemble')
    # ds = xr.concat(ds_list, dim='model_ensemble')

    nr = len(skill['perfect_model_ensemble'].data)
    global_ = region == 'GLOBAL'

    fn = 'Mapplots_CRPS_mean.png'
    plot_maps_mean(skill['CRPS'].mean('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Mean (N={nr}) relative CRPS change', cmap='PuOr_r')

    fn = 'Mapplots_Bias_mean.png'
    plot_maps_mean(skill['Bias'].mean('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Mean (N={nr}) relative Bias change')

    fn = 'Mapplots_RMSE_mean.png'
    plot_maps_mean(skill['RMSE'].mean('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Mean (N={nr}) relative RMSE change')

    fn = 'Mapplots_CRPS_median.png'
    plot_maps_mean(skill['CRPS'].median('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Median (N={nr}) relative CRPS change', cmap='PuOr_r')

    fn = 'Mapplots_Bias_median.png'
    plot_maps_mean(skill['Bias'].median('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Median (N={nr}) relative Bias change')

    fn = 'Mapplots_RMSE_median.png'
    plot_maps_mean(skill['RMSE'].median('perfect_model_ensemble'),
                   global_,
                   fn,
                   f'Median (N={nr}) relative RMSE change')

    fn = 'Mapplots_CRPS_p25.png'
    plot_maps_mean(skill['CRPS'].quantile(.25, 'perfect_model_ensemble'),
                   global_,
                   fn,
                   f'25 percentile (N={nr}) relative CRPS change', cmap='PuOr_r')


if __name__ == '__main__':
    main()
