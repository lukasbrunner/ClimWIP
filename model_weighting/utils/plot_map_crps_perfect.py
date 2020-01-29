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
Plot a map of the CRPS change between weighted and unweighted for a perfect
model setting.
"""
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from properscoring import crps_ensemble

from model_weighting.core.utils_xarray import area_weighted_mean

DATAPATH = '../../data/ModelWeighting/'
PLOTPATH = '../../plots/maps_crps/'
os.makedirs(PLOTPATH, exist_ok=True)


def crps_data(data, weights, model_ensemble=None, exclude_ensembles='all'):
    """Calculate the CRPS.

    Parameters
    ----------
    data : np.array, shape (N,)
    weights : np.array, shape (N,)
        Exactly one element of data has to be NaN. This will be used to
        identify the index of the perfect model.
    model_ensemble : np.array, shape (N,), optional
        List of model identifiers. Only needs to be given if exclude_ensembles
        is not 'none'.
    exclude_ensembles : {'all', 'same', 'none'}, optional
        - all: only use one ensemble member per model
        - same: use all ensemble members but exclude ensemble members from
          the same model as the perfect model
        - none: use all models and members except the perfect one

    Returns
    -------
    skill : float
    """
    assert np.isnan(weights).sum() == 1, 'exactly one weight has to be np.nan!'
    assert exclude_ensembles in ['all', 'same', 'none']
    if exclude_ensembles != 'none' and model_ensemble is None:
        raise ValueError('If exclude_ensembles is not none model_ensemble has to be given!')

    idx_perfect = np.where(np.isnan(weights))[0][0]
    if exclude_ensembles != 'none':
        models = [mm.split('_')[0] for mm in model_ensemble.data]
        if exclude_ensembles == 'all':
            _, idx_sel = np.unique(models, return_index=True)
            idx_sel = idx_sel[idx_sel != idx_perfect]
        elif exclude_ensembles == 'same':
            idx_sel = np.where(np.array(models) != models[idx_perfect])
        else:
            raise ValueError
    else:
        idx_sel = np.delete(np.arange(len(data)), idx_perfect)

    data_ref = data[idx_perfect]
    data_test = data[idx_sel]
    weights_test = weights[idx_sel]

    baseline = crps_ensemble(data_ref, data_test)
    weighted = crps_ensemble(data_ref, data_test, weights=weights_test)
    return (baseline - weighted) / baseline * 100.


def crps_xarray(ds, varn, exclude_ensembles='all'):
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
    skill = xr.apply_ufunc(
        crps_data, ds[varn], ds['weights'],
        input_core_dims=[['model_ensemble'], ['model_ensemble']],
        kwargs={'model_ensemble': ds['model_ensemble'],
                'exclude_ensembles': exclude_ensembles},
        vectorize=True,
    )
    return skill


def plot_map(skill, fn, how, global_):
    proj = ccrs.PlateCarree(central_longitude=0)
    fig, ax = plt.subplots(subplot_kw={'projection': proj})
    # fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.95)

    skill.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cbar_kwargs={'orientation': 'horizontal',
                     'pad': .1,
                     'label': '%'},
        robust=True,
        levels=11,
        center=0.,
        cmap='PuOr_r',
        extend='both',
    )

    ax.coastlines()

    longitude_formatter = LongitudeFormatter()
    latitude_formatter = LatitudeFormatter()
    if global_:
        ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
        ax.set_yticks(np.arange(-90, 91, 30), crs=proj)
    else:
        ax.set_xticks(np.arange(np.floor(skill['lon'].min()), skill['lon'].max(), 10), crs=proj)
        ax.set_yticks(np.arange(np.floor(skill['lat'].min()), skill['lat'].max(), 10), crs=proj)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_major_formatter(longitude_formatter)
    ax.yaxis.set_major_formatter(latitude_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.grid(True, zorder=-1)
    title = f'{how.title()} relative CRPS change (Global mean: {area_weighted_mean(skill).data:.1f})'
    ax.set_title(title)

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn.format(how=how), dpi=300)


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
    args = parser.parse_args()

    ds = xr.open_dataset(os.path.join(DATAPATH, args.filename))
    skill = crps_xarray(ds, ds.attrs['target'])

    if args.ext == 'show':
        fn = None
    else:
        fn = os.path.basename(args.filename).replace('.nc', '_{how}.' + args.ext)
        fn = os.path.join(PLOTPATH, f'map_{fn}')

    global_ = ds.attrs['region'] == 'GLOBAL'

    plot_map(skill.mean('perfect_model_ensemble'), fn, 'mean', global_)
    plot_map(skill.median('perfect_model_ensemble'), fn, 'median', global_)
    plot_map(skill.quantile(.25, 'perfect_model_ensemble'), fn, 'p25', global_)
    plot_map(skill.quantile(.75, 'perfect_model_ensemble'), fn, 'p75', global_)
    plot_map(skill.quantile(.1, 'perfect_model_ensemble'), fn, 'p10', global_)
    plot_map(skill.quantile(.9, 'perfect_model_ensemble'), fn, 'p90', global_)


if __name__ == '__main__':
    main()
