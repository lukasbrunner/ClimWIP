#!/usr/bin/env python3lkij
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-10-19 11:28:53 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: A collection of plot routines for quick checking of the output.

"""
import os
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from utils_python.xarray import add_hist

# matplotlib bug; issue 1120
# https://github.com/SciTools/cartopy/issues/1120
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

logger = logging.getLogger(__name__)


def plot_rmse(da, idx, cfg, da2=None):
    """
    Matrix plot of RMSEs for a given diagnostic.

    Parameters
    ----------
    da : xarray.DataArray, shape (N, N)
        Has to contain the RMSE from each model to each model
    varn : string
        A string for naming the plot (normally the name of the diagnostic)
    cfg : object
        Config object
    da2 : xarray.DataArray, shape (N,)
        Has to contain the RMSE from each model to the observations
    """
    if isinstance(idx, int):
        diagn = cfg.predictor_diagnostics[idx]
        agg = cfg.predictor_aggs[idx]
        title = 'Normalized RMSE {} {}'.format(diagn, agg)
        filename = 'rmse_{}-{}'.format(diagn, agg)
    else:
        title = 'Mean RMSE all diagnostics'
        filename = 'rmse_mean'
    path = os.path.join(cfg.plot_path, cfg.config)
    os.makedirs(path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 20))

    data = da.data
    xx = da['model_ensemble'].data
    if da2 is not None:
        data = np.concatenate([data, [da2.data]], axis=0)
        yy = list(xx) + ['Observations']
    else:
        yy = xx

    im = ax.matshow(data, vmin=0.)

    ax.set_xticks(np.arange(len(xx)))
    ax.set_yticks(np.arange(len(yy)))
    ax.set_xticklabels(xx)
    plt.xticks(rotation=90)
    ax.set_yticklabels(yy)
    ax.set_title(title, fontsize='xx-large', pad=70)

    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    filename = os.path.join(path, filename)
    plt.savefig(filename, dpi=300)
    plt.clf()
    logger.debug('Saved plot: {}.png'.format(filename))
    return filename


def plot_maps(ds, idx, cfg):
    """
    Mapplot of a given diagnostic and each model.

    Parameters
    ----------
    ds : xarray.Dataset
        Has to contain the varn and the dimensions 'lat', 'lon', and
        'model_ensemble'
    varn : string
        A string for naming the plot (normally the name of the diagnostic)
    cfg : object
        Config object
    """
    diagn = cfg.predictor_diagnostics[idx]
    agg = cfg.predictor_aggs[idx]
    syear = cfg.predictor_startyears[idx]
    eyear = cfg.predictor_endyears[idx]

    path = os.path.join(cfg.plot_path, cfg.config)
    os.makedirs(path, exist_ok=True)

    for model_ensemble in ds['model_ensemble'].data:
        proj = ccrs.PlateCarree(central_longitude=0)
        fig, ax = plt.subplots(subplot_kw={'projection': proj})
        ds.sel(model_ensemble=model_ensemble)[diagn].plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cbar_kwargs={'orientation': 'horizontal',
                         'label': 'tas (K)',
                         'pad': .1})
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS)
        xx, yy = np.meshgrid(ds['lon'], ds['lat'])
        ax.scatter(xx, yy, s=1, color='k')

        longitude_formatter = LongitudeFormatter()
        latitude_formatter = LatitudeFormatter()

        ax.set_xticks(np.arange(ds['lon'].min()-5, ds['lon'].max()+11, 10), crs=proj)
        ax.set_yticks(np.arange(ds['lat'].min()-5, ds['lat'].max()+11, 10), crs=proj)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_formatter(longitude_formatter)
        ax.yaxis.set_major_formatter(latitude_formatter)
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_title('{}: {} {} {}-{}'.format(model_ensemble, diagn, agg, syear, eyear))

        filename = os.path.join(path, 'map_{}-{}_{}'.format(diagn, agg, model_ensemble))
        plt.savefig(filename + '.png', dpi=300)
        plt.close()
        logger.debug('Saved plot: {}.png'.format(filename))
