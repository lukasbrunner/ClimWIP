#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Lukas Brunner, ETH Zurich

This file is part of ClimWIP.

ClimWIP is free software: you can redistribute it and/or modify
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
A collection of utility functions, which can either be applied directly to
xarray.Datasets or via xarray.apply_ufunc.
"""
import os
import logging
import warnings
import datetime
import subprocess
import numpy as np
import xarray as xr
import __main__ as main
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def area_weighted_mean_data(data, lat, lon):
    """Calculates an area-weighted average of data depending on latitude [and
    longitude] and handles missing values correctly.

    Parameters:
    - data (np.array): Data to be averaged, shape has to be (..., lat[, lon]).
    - lat (np.array): Array giving the latitude values.
    - lon (np.array): Array giving the longitude values (only used for
      consistency checks). lon can be None in which case data is considered
      as not containing the longitude dimension (hence latitude has to be the
      last dimension).

    Returns:
    Array of shape (...,) or float"""
    # input testing & pre-processing
    was_masked = False
    if isinstance(data, np.ma.core.MaskedArray):
        was_masked = True
    else:  # in case it a Python list
        data = np.array(data)
    lat = np.array(lat)
    assert len(lat.shape) == 1, 'lat has to be a 1D array'
    assert ((-90 <= lat) & (lat <= 90)).all(), 'lat has to be in [-90, 90]!'
    if lon is None:
        assert data.shape[-1] == len(lat), 'Axis -1 of data has to match lat!'
        data, lon = data.reshape(data.shape + (1,)), np.array([])
    else:
        lon = np.array(lon)
        assert len(lon.shape) == 1, 'lon has to be a 1D array'
        assert data.shape[-1] == len(lon), 'Axis -1 of data has to match lon!'
        assert data.shape[-2] == len(lat), 'Axis -2 of data has to match lat!'
        errmsg = 'lon has to be in [-180, 180] or [0, 360]!'
        assert (((-180 <= lon) & (lon <= 180)).all() or
                ((0 <= lon) & (lon <= 360)).all()), errmsg

    # create latitude weights and tile them to all longitudes, then flatten
    w_lat = np.cos(np.radians(lat))
    weights = np.tile(w_lat, (data.shape[-1], 1)).swapaxes(0, 1).ravel()
    # flatten lat-lon dimensions, mask missing values, average
    data_flat = np.ma.masked_invalid(data.reshape(data.shape[:-2] + (-1,)))
    mean = np.ma.average(data_flat, axis=-1, weights=weights)

    # NOTE: if data is a single value and not invalid it will be a normal
    # float at this point and no longer masked!
    if isinstance(mean, float):
        return mean
    elif was_masked:  # if input was masked array also return a masked array
        return mean
    return mean.filled(fill_value=np.nan)


def area_weighted_mean(
        ds, latn=None, lonn=None, keep_attrs=True, suppress_warning=False):
    """xarray version of utils_python.physics.area_weighed_mean

    Parameters
    ----------
    ds : {xarray.Dataset, xarray.DataArray}
        Has to contain at least latitude and longitude dimensions
    lonn : string, optional
    latn : string, optional
    keep_attrs : bool, optional
        Whether to keep the global attributes
    suppress_warning : bool, optional
        Suppress warnings about nan-only instances

    Returns
    -------
    mean : same type as input with (lat, lon) dimensions removed
    """
    if suppress_warning:
        warnings.simplefilter('ignore')
    if latn is None and 'lat' in ds.dims:
        latn = 'lat'
    elif latn is None:
        raise ValueError
    if lonn is None and 'lon' in ds.dims:
        lonn = 'lon'
    elif latn is None:
        raise ValueError

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        ds = ds.to_dataset(name='data')

    ds_mean = ds.mean((latn, lonn), keep_attrs=keep_attrs)  # create this just to fill
    for varn in set(ds.variables).difference(ds.dims):
        if latn in ds[varn].dims and lonn in ds[varn].dims:
            var = ds[varn].data
            axis_lat = ds[varn].dims.index(latn)
            axis_lon = ds[varn].dims.index(lonn)
            var = np.moveaxis(var, (axis_lat, axis_lon), (-2, -1))
            mean = area_weighted_mean_data(var, ds[latn].data, ds[lonn].data)
        elif latn in ds[varn].dims:
            var = ds[varn].data
            axis_lat = ds[varn].dims.index(latn)
            var = np.moveaxis(var, axis_lat, -1)
            mean = area_weighted_mean_data(var, ds[latn].data)
        elif lonn in ds[varn].dims:
            mean = ds[varn].mean(lonn).data
        else:
            continue

        if '_FillValue' in ds[varn].encoding.keys():
            fill_value = ds[varn].encoding['_FillValue']
            if fill_value != np.nan:
                mean = np.where(np.isnan(mean), fill_value, mean)
        ds_mean[varn].data = mean

    warnings.resetwarnings()
    if was_da:
        return ds_mean['data']
    return ds_mean


def add_revision(ds, warning=True, attr='history'):
    """
    Add script name to the netCDF history attribute similar to cdo.

    Parameters
    ----------
    ds : xarray.Dataset
    warning : bool, optional
        Log a warning if the repository is not under Git control.

    Returns
    -------
    ds : same as input

    Format
    ------
    Git:
    time_stamp git@url:git_rep.git /relative/path/script.py, tag: ##, branch: branch_name

    time_stamp :  The current time
    git@url:git_rep.git : Origin of the git reposiroty
        git config --get remote.origin.url
    /relative/path/script.py : relative path
        git ls-tree --full-name --name-only HEAD
    ## : revision hash or revision hash
        git describe --always
    branch_name : currently checked out branch
        git rev-parse --abbrev-ref HEAD
    If the main calling script checked in a git repository 'git@url:git_rep.git' like
    (branch_name)~ ./code/script.py

    Not in Git:
    'time_stamp /absolute/path/script.py'
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = str(os.path.realpath(main.__file__))
    try:
        revision = subprocess.check_output([
            'git', 'describe', '--always']).strip().decode()
        rep_name = subprocess.check_output([
            'git', 'config', '--get', 'remote.origin.url']).strip().decode()
        branch_name = subprocess.check_output([
            'git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode()
        filename_git = subprocess.check_output([
            'git', 'ls-tree', '--full-name',
            '--name-only', 'HEAD', filename]).strip().decode()
        if filename_git == '':
            if warning:
                logmsg = '{} is in a Git repository but not checked in!'.format(
                    filename)
                logger.warning(logmsg)
            str_add = '{} {}'.format(time, filename)
        else:
            str_add = '{} {}: {}, revision hash: {}, branch: {}'.format(
                time, rep_name, filename_git, revision, branch_name)
    except subprocess.CalledProcessError:
        if warning:
            logmsg = '{} is not under Git control!'.format(filename)
            logger.warning(logmsg)
        str_add = '{} {}'.format(time, filename)

    if attr in ds.attrs:
        ds.attrs[attr] = '{} {}'.format(ds.attrs[attr], str_add)
    else:
        ds.attrs[attr] = str_add
    return ds


def weighted_distance_matrix(data, lat=None):
    """An area-weighted RMS full distance matrix"""
    if lat is None:
        w_lat = np.ones(data.shape[-2])
    else:
        w_lat = np.cos(np.radians(lat))
    weights = np.tile(w_lat, (data.shape[-1], 1)).swapaxes(0, 1).ravel()
    data = data.reshape((data.shape[0], weights.shape[0]))

    # only select grid points which are not nan for all models
    idx = np.where(np.all(np.isfinite(data), axis=0))[0]
    data = data[:, idx]
    weights = weights[idx]

    # normalize (!)
    weights /= weights.sum()

    d_matrix = squareform(pdist(data, metric='euclidean', w=weights))
    np.fill_diagonal(d_matrix, np.nan)
    return d_matrix


def distance_uncertainty(var, obs_min, obs_max):
    """Account for uncertainties in the observations by setting
    distances within the observational spread to zero"""
    lower = var < obs_min
    higher = var > obs_max
    between = (var >= obs_min) & (var <= obs_max)
    diff = np.zeros_like(var) * np.nan
    diff[lower] = var[lower] - obs_min[lower]
    diff[higher] = var[higher] - obs_max[higher]
    diff[between] = 0.
    # NaN in either array results in NaN in diff
    return diff


def detrend(data):
    if np.any(np.isnan(data)):
        return data * np.nan
    return signal.detrend(data)


def trend(data):
    if np.any(np.isnan(data)):
        return np.nan
    xx = np.arange(len(data))
    return stats.linregress(xx, data).slope


def correlation(arr1, arr2):
    if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
        return np.nan
    return stats.pearsonr(arr1, arr2)[0]



def _antimeridian_pacific(ds, lonn):
    """Returns True if the antimeridian is in the Pacific (i.e. longitude runs
    from -180 to 180."""
    if lonn is None:
        lonn = get_longitude_name(ds)
    if ds[lonn].min() < 0 or ds[lonn].max() < 180:
        return True
    return False


def flip_antimeridian(ds, to='Pacific', lonn=None):
    """
    Flip the antimeridian (i.e. longitude discontinuity) between Europe
    (i.e., [0, 360)) and the Pacific (i.e., [-180, 180)).

    Parameters:
    - ds (xarray.Dataset or .DataArray): Has to contain a single longitude
      dimension.
    - to='Pacific' (str, optional): Flip antimeridian to one of
      * 'Europe': Longitude will be in [0, 360)
      * 'Pacific': Longitude will be in [-180, 180)
    - lonn=None (str, optional): Name of the longitude dimension. If None it
      will be inferred by the CF convention standard longitude unit.

    Returns:
    same type as input ds
    """
    if lonn is None and 'lon' in ds.dims:
        lonn = 'lon'
    else:
        raise ValueError

    attrs = ds[lonn].attrs

    if to.lower() == 'europe' and not _antimeridian_pacific(ds, lonn):
        return ds  # already correct, do nothing
    elif to.lower() == 'pacific' and _antimeridian_pacific(ds, lonn):
        return ds  # already correct, do nothing
    elif to.lower() == 'europe':
        ds = ds.assign_coords(**{lonn: (ds.lon % 360)})
    elif to.lower() == 'pacific':
        ds = ds.assign_coords(**{lonn: (((ds.lon + 180) % 360) - 180)})
    else:
        errmsg = 'to has to be one of [Europe | Pacific] not {}'.format(to)
        raise ValueError(errmsg)

    was_da = isinstance(ds, xr.core.dataarray.DataArray)
    if was_da:
        da_varn = ds.name
        enc = ds.encoding
        ds = ds.to_dataset()

    idx = np.argmin(ds[lonn].data)
    varns = [varn for varn in ds.variables if lonn in ds[varn].dims]
    for varn in varns:
        if xr.__version__ > '0.10.8':
            ds[varn] = ds[varn].roll(**{lonn: -idx}, roll_coords=False)
        else:
            ds[varn] = ds[varn].roll(**{lonn: -idx})

    ds[lonn].attrs = attrs
    if was_da:
        da = ds[da_varn]
        da.encoding = enc
        return da
    return ds


def quantile(data, quantiles, weights=None, interpolation='linear',
             old_style=False):
    """Calculates weighted quantiles.

    Parameters:
    - data (np.array): Array of data (N,)
    - quantiles (np.array): Array of quantiles (M,) in [0, 1]
    - weights=None (np.array, optional): Array of weights (N,)
    - interpolation='linear' (str, optional): String giving the interpolation
      method (equivalent to np.percentile). "This optional parameter specifies
      the interpolation method to use when the desired quantile lies between
      two data points." One of (with i < j):
      * linear: i + (j - i) * fraction where fraction is the fractional part
        of the index surrounded by i and j
      * lower: i  NOTE: might lead to unexpected results for integers (see
        tests/test_math.test_quantile_interpolation)
      * higher: j  NOTE: might lead to unexpected results for integers
      * nearest: i or j whichever is nearest
      * midpoint: (i + j) / 2. TODO: not yet implemented!
    - old_style=False (bool, optional): If True, will correct output to be
      consistent with np.percentile.

    Returns:
    np.array of shape (M,)"""
    data = np.array(data)
    quantiles = np.array(quantiles)
    if np.any(np.isnan(data)):
        errmsg = ' '.join([
            'This function is not tested with missing data! Comment this test',
            'if you want to use it anyway.'])
        raise ValueError(errmsg)
    if data.ndim != 1:
        errmsg = 'data should have shape (N,) not {}'.format(data.shape)
        raise ValueError(errmsg)
    if np.any(quantiles < 0.) or np.any(quantiles > 1.):
        errmsg = 'quantiles should be in [0, 1] not {}'.format(quantiles)
        raise ValueError(errmsg)
    if weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.array(weights)
        if data.shape != weights.shape:
            errmsg = ' '.join([
                'weights need to have the same shape as data ',
                '({} != {})'.format(weights.shape, data.shape)])
            raise ValueError(errmsg)
        # remove values with weights zero
        idx = np.where(weights == 0)[0]
        weights = np.delete(weights, idx)
        data = np.delete(data, idx)

    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - .5*weights

    if old_style:  # consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:  # more correct (see reference for a discussion)
        weighted_quantiles /= np.sum(weights)

    results = np.interp(quantiles, weighted_quantiles, data)

    if interpolation == 'linear':
        return results
    elif interpolation == 'lower':
        if isinstance(results, float):
            return data[data <= results][-1]
        return np.array([data[data <= rr][-1] for rr in results])
    elif interpolation == 'higher':
        if isinstance(results, float):
            return data[data >= results][0]
        return np.array([data[data >= rr][0] for rr in results])
    elif interpolation == 'nearest':
        if isinstance(results, float):
            return data[np.argmin(np.abs(data - results))]
        return np.array([data[np.argmin(np.abs(data - rr))] for rr in results])
    elif interpolation == 'midpoint':
        raise NotImplementedError
    else:
        errmsg = ' '.join([
            'interpolation has to be one of [linear | lower | higher |',
            'nearest | midpoint] and not {}'.format(interpolation)])
        raise ValueError(errmsg)
