#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Lukas Brunner, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Authors
-------
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract
--------
Plot a map with the difference between the weighted and unweighted mean.
"""
import os
import warnings
import argparse
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

DATAPATH = '../../data/ModelWeighting/'
PLOTPATH = '../../plots/maps/'


def get_plot_config(varn, region):
    if varn == 'tas':
        return (
            dict(
                cmap='RdBu_r',
                extend='both',
                center=0,
                robust=True,
                levels=11
                # vmin=-.6,
                # vmax=.6,
                # levels=np.concatenate((np.arange(-.6, 0, .1), np.arange(.1, .61, .1))),

            ),
            '\n'.join([f'{region} temperature difference (K)',
                       'hatching: 90% significance (1000 member bootstrap)']))
    elif varn == 'pr':
        return (
            dict(
                cmap='BrBG',
                extend='both',
                center=0,
                robust=True,
                levels=11
            ),
            '\n'.join([f'{region} relative precipitation difference (%)',
                       'hatching: 90% significance (1000 member bootstrap)']))
    return (
        {},
        '\n'.join([f'{region} {varn}',
                   'hatching: 90% significance (1000 member bootstrap)']))


def preprocess(ds, varn, relative=False):
    with warnings.catch_warnings():
        # suppress warnings on masked ocean grid cells
        warnings.filterwarnings('ignore')
        av = ds[varn].mean('model_ensemble')
        avw = xr.apply_ufunc(
            np.average, ds[varn],
            input_core_dims=[['model_ensemble']],
            kwargs={'weights': ds['weights']},
            vectorize=True)
    av_diff = avw - av
    if relative:
        av_diff /= 0.01 * av
    ds['mean'] = av
    ds['wmean'] = avw
    ds['difference'] = av_diff


def bootstrap(ds, varn, percentile=(.05, .95), relative=False):
    diffs = []
    weights = ds['weights'].copy().data
    for idx in range(1000):
        np.random.RandomState(idx).shuffle(weights)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            avw = xr.apply_ufunc(
                np.average, ds[varn],
                input_core_dims=[['model_ensemble']],
                kwargs={'weights': weights},
                vectorize=True)
        av_diff = avw - ds['mean']
        if relative:
            av_diff /= 0.01 * ds['mean']
        diffs.append(av_diff)

    ds_diff = xr.concat(diffs, dim='realizations')
    lower, upper = ds_diff.quantile(percentile, 'realizations')
    ds['significant'] = (ds['difference'] < lower) | (ds['difference'] > upper)


# copied form plot_utils
def hatching(ax, lat, lon, condition, hatch='/////', force=False, wrap_lon=False):
    """Adds a hatching layer to an axis object.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis object
    lat : sequence of float, shape (M,)
    lon : sequence of float, shape (N,)
    condition : sequence of bool, shape (M, N)
        Hatching will be drawn where condition is True.
    hatch : valid hatch string
        Note that multiple equivalent characters will lead to a finer hatching.
    force : bool, optional
        If True also work with unevenly spaced lat and lon. This might lead to
        unexpected behavior if the gird is too uneven.
    wrap_lon : bool, optional
        Wrap longitude to [-180, 180).

    Returns
    -------
    None"""
    if isinstance(lat, xr.core.dataarray.DataArray):
        lat = lat.data
    if isinstance(lon, xr.core.dataarray.DataArray):
        lon = lon.data

    dlat = np.unique(lat[1:] - lat[:-1])
    dlon = np.unique(lon[1:] - lon[:-1])

    if force:
        dlat = [np.mean(dlat)]
        dlon = [np.mean(dlon)]

    assert len(dlat) == 1, 'must be evenly spaced'
    assert len(dlon) == 1, 'must be evenly spaced'
    dxx = dlon[0] / 2
    dyy = dlat[0] / 2
    assert np.shape(condition) == (len(lat), len(lon))
    ii, jj = np.where(condition)
    lat_sel = lat[ii]
    lon_sel = lon[jj]

    if wrap_lon:
        lon_sel = [ll if ll < 180 else ll - 360 for ll in lon_sel]

    patches = [
        Polygon(
            [[xx-dxx, yy+dyy], [xx-dxx, yy-dyy],
             [xx+dxx, yy-dyy], [xx+dxx, yy+dyy]])
        for xx, yy in zip(lon_sel, lat_sel)]
    pp = PatchCollection(patches)
    pp.set_alpha(0.)
    pp.set_hatch(hatch)
    ax.add_collection(pp)


def plot_map(ds, title, kwargs, global_, filename=None):
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(subplot_kw={'projection': proj})

    ds['difference'].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': ''},
        kwargs={'zorder': 100},
        **kwargs,
    )

    ax.coastlines()

    longitude_formatter = LongitudeFormatter()
    latitude_formatter = LatitudeFormatter()
    if global_:
        ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
        ax.set_yticks(np.arange(-90, 91, 30), crs=proj)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_formatter(longitude_formatter)
        ax.yaxis.set_major_formatter(latitude_formatter)
    else:
        ax.set_xticks(np.arange(np.floor(ds['lon'].min()), ds['lon'].max(), 10), crs=proj)
        ax.set_yticks(np.arange(np.floor(ds['lat'].min()), ds['lat'].max(), 10), crs=proj)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_formatter(longitude_formatter)
        ax.yaxis.set_major_formatter(latitude_formatter)

    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.grid(True, zorder=-1)
    ax.set_title(title)

    if 'significant' in ds:
        hatching(ax, ds['lat'], ds['lon'], ds['significant'], force=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='filename', type=str,
        help='A valid filename as produced by model_weighting_main.py')
    parser.add_argument(
        '--plot-type', '-t', dest='ext', default='png', type=str,
        help=' '.join([
            'A valid plot extension specifiying the file type. A special case',
            'is "show" which will call plt.show() instead of saving']))
    parser.add_argument(
        '--no-hatching', '-nh', dest='hatch', action='store_false',
        help='')
    args = parser.parse_args()

    ds = xr.open_dataset(os.path.join(DATAPATH, args.filename))

    varn = ds.attrs['target']
    region = ds.attrs['region']
    relative = True if varn in ['pr'] else False

    preprocess(ds, varn, relative=relative)
    if args.hatch:
        bootstrap(ds, varn, relative=relative)
    plot_kwargs, title = get_plot_config(varn, region)
    plot_map(ds, title, plot_kwargs, region == 'GLOBAL')

    if args.ext == 'show':
        plt.show()
    else:
        os.makedirs(PLOTPATH, exist_ok=True)
        plot_filename = os.path.basename(args.filename).replace(
            '.nc', f'.{args.ext}')
        filename = os.path.join(PLOTPATH, f'map_{plot_filename}')
        plt.savefig(filename, dpi=300)
        print(f'Saved: {filename}')


if __name__ == '__main__':
    main()
