#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import regionmask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
import seaborn as sns
from glob import glob

from utils_python.xarray import area_weighted_mean, flip_antimeridian
from model_weighting.core.utils import read_config, log_parser

from boxplot import boxplot

SAVEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../../plots/boxplots_skill'
os.makedirs(SAVEPATH, exist_ok=True)


def read_input():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filename_patterns',
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--path', '-p', dest='path', type=str, default='',
        help='')
    parser.add_argument(
        '--title', '-t', dest='title', type=str, default=None,
        help='')
    parser.add_argument(
        '--labels', '-l', dest='labels', default=None,
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--savename', '-s', dest='savename', type=str, default=None,
        help='')
    parser.add_argument(
        '--ylim', dest='ylim', default=None,
        type=lambda x: x.split(', '),
        help='')
    parser.add_argument(
        '--mean-of-crps', '-crps2', dest='mean_of_crps', action='store_true',
        help='')
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.filenames):
        logmsg = '--labels needs to have same length as filenames! Falling back to default'
        args.labels = None
        print(logmsg)
    return args


def get_skill(da, da_obs, weights, mean_of_crps):
    if mean_of_crps:
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

    return (crps_baseline - crps_weighted) / crps_baseline * 100


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
        try:
            filename = os.path.join(obs_path, '{}_mon_{}_g025_future.nc'.format(cfg.target_diagnostic, obs_id))
            ds_var2 = xr.open_dataset(filename, use_cftime=True).load()[cfg.target_diagnostic]
            ds_var = xr.concat([ds_var, ds_var2], dim='time')
        except FileNotFoundError:
            pass

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
        land_mask = regionmask.defined_regions.natural_earth.land_110.mask(ds_var, wrap_lon=180) == 1
        ds_var = ds_var.where(land_mask)

    return ds_var.squeeze()


def main():
    args = read_input()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, right=.99, bottom=.22, top=.91)

    xticks = []
    xticklabels = []
    for xx, filename_pattern in enumerate(args.filename_patterns):
        filename_pattern = os.path.join(args.path, filename_pattern)
        filenames = glob(filename_pattern)
        skill_list = []
        for filename in filenames:
            ds = xr.open_dataset(filename)
            varn = ds.attrs['target']
            cfg = read_config(ds.attrs['config'], ds.attrs['config_path'])
            # log_parser(cfg)
            ds_obs = read_obs(ds, cfg)
            skill = get_skill(ds[varn], ds_obs, ds['weights'], args.mean_of_crps)
            skill_list.append(skill)

        skill = xr.concat(skill_list, dim='perfect_model_ensemble')

        if args.mean_of_crps:
            median = float(area_weighted_mean(skill.median('perfect_model_ensemble')).data)
            mean = float(area_weighted_mean(skill.mean('perfect_model_ensemble')).data)
            p25, p75 = area_weighted_mean(skill.quantile((.25, .75), 'perfect_model_ensemble')).data
            p05, p95 = area_weighted_mean(skill.quantile((.05, .95), 'perfect_model_ensemble')).data
        else:
            median = float(skill.median('perfect_model_ensemble').data)
            mean = float(skill.mean('perfect_model_ensemble').data)
            p25, p75 = skill.quantile((.25, .75), 'perfect_model_ensemble').data
            p05, p95 = skill.quantile((.05, .95), 'perfect_model_ensemble').data

        boxplot(
            ax, xx,
            mean=mean,
            median=median,
            box=(p25, p75),
            whis=(p05, p95),
            width=.8,
            alpha=1,
            color=sns.xkcd_rgb['greyish'],
            # whis_kwargs={'caps_width': .6},
            # median_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
            # mean_kwargs={'linestyle': '-', 'color': 'k', 'linewidth': .5},
        )

        xticks.append(xx)
        xticklabels.append(ds.attrs['config'])

    ax.grid(axis='y')
    ax.axhline(0, color='k', zorder=2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha='right')

    ax.set_ylim(-100, 100)
    ax.set_ylabel('Relative CRPS change (%)', labelpad=-3)

    plt.title(f'Change in CRPS')

    plt.savefig(os.path.join(SAVEPATH, 'test_crps_pseudo_obs.png'), dpi=300)


if __name__ == '__main__':
    main()
