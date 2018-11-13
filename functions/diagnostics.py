#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-11-13 15:38:33 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import logging
import warnings
import regionmask
import numpy as np
import xarray as xr
import salem
from cdo import Cdo
cdo = Cdo()
from scipy.stats import pearsonr

from utils_python.decorators import vectorize

from utils_python.xarray import select_region, flip_antimeridian

logger = logging.getLogger(__name__)
REGION_DIR = '{}/../cdo_data/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'  # 'seamask_g025.nc'
unit_save = None


def calculate_net_radiation(infile, varns, diagn):
    assert varns == ('rlds', 'rlus', 'rsds', 'rsus')
    da1 = xr.open_dataset(infile, decode_cf=False)[vanrs[0]]
    da2 = xr.open_dataset(infile.replace(varns[0], varns[1]), decode_cf=False)[vanrs[1]]
    da3 = xr.open_dataset(infile.replace(varns[0], varns[2]), decode_cf=False)[vanrs[2]]
    da4 = xr.open_dataset(infile.replace(varns[0], varns[3]), decode_cf=False)[vanrs[3]]

    da = (da1-da2) + (da3-da4)
    # TODO: units; positive direction definition as attrs
    ds = da.to_dataset(name=diagn)
    ds.to_netcdf(outname)


def standardize_units(da, varn):
    """Convert units to a common standard"""
    if 'units' in da.attrs.keys():
        unit = da.attrs['units']
        attrs = da.attrs
    else:
        logmsg = 'units attribute not found for {}'.format(varn)
        logger.warning(logmsg)
        return None

    # --- precipitation ---
    if varn == 'pr':
        newunit = 'mm/day'
        if unit == 'kg m-2 s-1':
            da.data *= 24*60*60
            da.attrs = attrs
            da.attrs['units'] = newunit
        elif unit == newunit:
            pass
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)

    # --- temperature ---
    elif varn in ['tas', 'tasmax', 'tasmin', 'tos']:
        newunit = "degC"
        if unit == 'K':
            da.data -= 273.15
            da.attrs = attrs
            da.attrs['units'] = newunit
        elif unit.lower() in ['degc', 'deg_c', 'celsius', 'degreec',
                              'degree_c', 'degree_celsius']:
            # https://ferret.pmel.noaa.gov/Ferret/documentation/udunits.dat
            da.attrs['units'] = newunit
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)
    else:
        logmsg = 'Variable {} not covered in standardize_units'.format(varn)
        logger.warning(logmsg)

    return da


def calculate_basic_diagnostic(infile, varn,
                               outfile=None,
                               time_period=None,
                               season=None,
                               time_aggregation=None,
                               mask_ocean=False,
                               region='GLOBAL',
                               overwrite=False,
                               regrid=False):
    """
    Calculate a basic diagnostic from a given file.

    A basic diagnostic calculated from a input file by selecting a given
    region, time period, and season as well as applying a land-sea mask.
    Also, the time dimension is aggregated by different methods.

    Parameters
    ----------
    infile : str
        Full path of the input file. Must contain exactly one non-dimension
        variable.
    varn : str
        The variable contained in infile.
    outfile : str
        Full path of the output file. Path must exist.
    time_period : tuple of two strings, optional
        Start and end of the time period. Each string must be in the form
        {"yyyy", "yyyy-mm", "yyyy-mm-dd"}.
    season : {'JJA', 'SON', 'DJF', 'MAM', 'ANN'}, optional
    time_aggregation : {'CLIM', 'STD', 'TREND'}, optional
        Type of time aggregation to use.
    mask_ocean : bool, optional
    region : list of strings or str, optional
        Each string must be a valid SREX region
    overwrite : bool, optional
        If True overwrite existing outfiles otherwise read and return them.
    regrid : bool, optional
        If True the file will be regridded by cdo.remapbic before opening.

    Returns
    -------
    diagnostic : xarray.DataArray
    """
    if not overwrite and os.path.isfile(outfile):
        logger.debug('Diagnostic already exists & overwrite=False, skipping.')
        return xr.open_dataset(outfile)

    if regrid:
        infile = cdo.remapbic(os.path.join(REGION_DIR, MASK),
                                options='-b F64', input=infile)

    da = xr.open_dataarray(infile)
    da = flip_antimeridian(da, to='Pacific')
    assert da.name == varn
    assert np.all(da['lat'].data == np.arange(-88.75, 90., 2.5))
    assert np.all(da['lon'].data == np.arange(-178.75, 180., 2.5))

    if time_period is not None:
        da = da.sel(time=slice(str(time_period[0]), str(time_period[1])))

    if season in ['JJA', 'SON', 'DJF', 'MAM']:
        da = da.isel(time=da['time.season']==season)
    elif season == 'ANN':
        pass
    else:
        raise NotImplementedError('season={}'.format(season))

    if region != 'GLOBAL':
        if isinstance(region, str):
            region = [region]
        masks = []
        keys = regionmask.defined_regions.srex.map_keys(region)
        for key in keys:
            masks.append(
                regionmask.defined_regions.srex.mask(da) == key)
        mask = sum(masks) > 0
        da = da.where(mask)
        # NOTE: we could also use da.salem.roi here.
        # salem.roi is super flexible, taking corner points, polygons, and shape files
        # cut the smallest rectangular region containing all unmasked grid points
        da = da.salem.subset(roi=da.isel(time=0))

    if mask_ocean:
        sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(da) == 0
        da = da.where(sea_mask)

    da = standardize_units(da, varn)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')

        if time_aggregation == 'CLIM':
            da = da.groupby('time.year').mean('time')
            da = da.mean('year')
        elif time_aggregation == 'STD':
            da = xr.apply_ufunc(scipy.detrend, da,
                                input_core_dims=[['time']],
                                output_core_dims=['time'],
                                keep_attrs=True)
            da = da.std('time')
        elif time_aggregation == 'TREND':
            da = xr.apply_ufunc(scipy.linregress, da,
                                input_core_dims=[['time']],
                                output_core_dims=['time'],
                                keep_attrs=True)
        elif time_aggregation is None or time_aggregation == 'CORR':
            pass
        else:
            NotImplementedError('time_aggregation={}'.format(time_aggregation))

    ds = da.to_dataset(name=varn)
    if outfile is not None:
        ds.to_netcdf(outfile)
    return ds


def calculate_diagnostic(infile, diagn, base_path, **kwargs):
    """
    Calculate basic or derived diagnostics depending on input.

    Parameters
    ----------
    infile : str
        Full path of the input file. Must contain exactly one non-dimension
        variable.
    diagn : str or dict
        * if str: diagn is assumed to also be a basic variable and will
          directly be used to call calculate_basic_diagnostic
        * if dict: diagn has to be exactly one key-value pair with the values
          representing basic variables and the key representing the name of
          the newly created diagnostic (e.g., {'tasclt': ['tas', clt']} will
          calculate the correlation between tas and clt.
    base_path : str
        The path in which to save the calculated diagnostic file.
    kwargs : dict
        Keyword arguments passed on to calculate_basic_diagnostic.

    Returns
    -------
    diagnostic : xarray.DataArray
    """
    def get_outfile(**kwargs):
        kwargs['infile'] = os.path.basename(kwargs['infile']).replace('.nc', '')
        if isinstance(kwargs['region'], list):
            kwargs['region'] = '-'.join(kwargs['region'])

        return os.path.join(base_path, '_'.join([
            '{infile}_{time_period[0]}-{time_period[1]}_{season}',
            '{time_aggregation}_{region}.nc']).format(**kwargs))

    @vectorize('(n),(n)->()')
    def _corr(arr1, arr2):
        return pearsonr(arr1, arr2)[0]

    if isinstance(diagn, str):  # basic diagnostic
        outfile = get_outfile(infile=infile, **kwargs)
        return calculate_basic_diagnostic(infile, diagn, outfile, **kwargs)
    elif isinstance(diagn, dict):  # derived diagnostic
        assert len(diagn.keys()) == 1
        key = list(diagn.keys())[0]
        varns = diagn.pop(key)  # basic variables
        diagn = key  # derived diagnostic

        if diagn == 'rnet':
            tmpfile = os.path.join(
                base_path, os.path.basename(infile).replace(varns[0], diagn))
            ds = calculate_net_radiation(infile, varns, tmpfile, diang)
            outfile = get_outfile(infile=tmpfile, **kwargs)
            return calculate_basic_diagnostic(tmpfile, diagn, outfile, **kwargs)
        elif kwargs['time_aggregation'] == 'CORR':
            assert len(varns) == 2, 'can only correlate two variables'
            assert varns[0] != varns[1], 'can not correlate same variables'
            outfile1 = get_outfile(infile=infile, **kwargs)
            ds1 = calculate_basic_diagnostic(infile, varns[0], outfile1, **kwargs)

            infile2 = infile.replace(varns[0], varns[1])
            outfile2 = get_outfile(infile=infile2, **kwargs)
            ds2 = calculate_basic_diagnostic(infile2, varns[1], outfile2, **kwargs)
            da = xr.apply_ufunc(_corr, ds1[varns[0]], ds2[varns[1]],
                                input_core_dims=[['time'], ['time']],
                                vectorize=True)
            return da.to_dataset(name=diagn)
