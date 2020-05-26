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
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch
- Anna Merrifield || anna.merrifield@ethz.ch
- Mario S. Koenz || mskoenz@gmx.net

Abstract
--------

"""
import os
import re
import sys
import logging
import warnings
import regionmask
import numpy as np
import xarray as xr
from cdo import Cdo

from .utils_xarray import (
    detrend,
    trend,
    correlation,
    flip_antimeridian,
    area_weighted_mean,
)

cdo = Cdo()
logger = logging.getLogger(__name__)

REGION_DIR = '{}/../shapefiles/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'


def calculate_net_radiation(infile, varns, outname, diagn):
    assert varns == ('rlds', 'rlus', 'rsds', 'rsus')
    da1 = xr.open_dataset(infile, decode_cf=False)[varns[0]]
    da2 = xr.open_dataset(infile.replace(varns[0], varns[1]), decode_cf=False)[varns[1]]
    da3 = xr.open_dataset(infile.replace(varns[0], varns[2]), decode_cf=False)[varns[2]]
    da4 = xr.open_dataset(infile.replace(varns[0], varns[3]), decode_cf=False)[varns[3]]

    da = (da1-da2) + (da3-da4)
    da.attrs['units'] = da1.units
    try:
        da.attrs['_FillValue'] = da1._FillValue
    except AttributeError:
        da.attrs['_FillValue'] = 1e20
    da.attrs['long_name'] = 'Surface Downwelling Net Radiation'
    da.attrs['standard_name'] = 'surface_downwelling_net_flux_in_air'
    # TODO: units; positive direction definition as attrs
    ds = da.to_dataset(name=diagn)

    ds.to_netcdf(outname)


def standardize_units(da, varn):
    """Convert units to a common standard"""
    if 'units' in da.attrs.keys():
        unit = da.attrs['units']
    else:
        logmsg = 'units attribute not found for {}'.format(varn)
        logger.warning(logmsg)
        return None

    # --- precipitation ---
    if varn == 'pr':
        newunit = 'mm/day'
        if unit == 'kg m-2 s-1':
            da.data *= 24*60*60
            da.attrs['units'] = newunit
        elif unit in ['m', 'mm']:
            errmsg = '\n'.join([
                'The use of m & mm is highly problematic in monthly files!',
                'There correct interpretation (in my opinion) is as',
                'precipitation sums over the respective month. But very often',
                'they actually represent the mean of daily m or mm and should',
                'therefore have the unit m/day & mm/day. This is even wrong',
                'in the ERA5 monthly mean files downloaded from copernicus.',
                # 'To avoid having this mistake fail silently and apply a',
                # 'wrong correction here the usage of m & mm as unit is',
                # 'disallowed for now!
                'Please check the input files and change',
                'the unit accordingly (e.g., using',
                '<cdo chunit,m,m/day infile outfile>'])
            # # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#calendar
            # if da['time'].encoding['calendar'] in ['gregorian', 'standard', 'proleptic_gregorian']:
            #     days_in_month = xr.CFTimeIndex.to_datetimeindex(da.time).days_in_month
            #     days_in_month = xr.DataArray(days_in_month, coords={'time': da.time}, dims='time')
            # elif da['time'].encoding['calendar'] in ['noleap', '365_day']:
            #     days_in_month = 365
            # elif da['time'].encoding['calendar'] in ['all_leap', '366_day']:
            #     days_in_month = 366
            # elif da['time'].encoding['calendar'] in ['360_day']:
            #     days_in_month = 360
            # else:
            #     raise ValueError('Could not calculate days in month but need it for unit conversion!')
            # da /= days_in_month
            # if unit == 'm':
            #     da.data *= 1000
            # raise ValueError(errmsg)
            logger.warning(errmsg)
        elif unit == 'm/day':  # ERA5
            da.data *= 1000
            da.attrs['units'] = newunit
        elif unit == newunit:
            pass
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)

    # --- temperature ---
    elif varn in ['tas', 'tasmax', 'tasmin', 'tos']:
        newunit = "degC"
        if unit == newunit:
            pass
        elif unit in ['K', 'Kelvin']:
            da.data -= 273.15
            da.attrs['units'] = newunit
        elif unit.lower() in ['degc', 'deg_c', 'celsius', 'degreec',
                              'degree_c', 'degree_celsius']:
            # https://ferret.pmel.noaa.gov/Ferret/documentation/udunits.dat
            da.attrs['units'] = newunit
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)

    # --- pressure ---
    elif varn in ['psl']:
        newunit = 'Pa'
        if unit == newunit:
            pass
        elif unit == 'pa':
            da.attrs['units'] = newunit
        elif unit in ['hPa', 'hpa']:
            da.data *= 100.
            da.attrs['units'] = newunit
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)

    # --- radiation ---
    elif varn in ['rsds', 'rsus', 'rlds', 'rlus', 'rnet']:
        newunit = 'W m**-2'
        if unit == newunit:
            pass
        elif unit == 'W m-2':
            da.attrs['units'] = newunit
        elif unit in ['J m**-2', 'J m-2']:
            errmsg = '\n'.join([
                'The use of J is highly problematic in monthly files!',
                'There correct interpretation (in my opinion) is as',
                'Energy sums over the respective month. But very often',
                'they actually represent the mean of daily J and should',
                'therefore have the unit J/day. This is even wrong',
                'in the ERA5 monthly mean files downloaded from copernicus.',
                'To avoid having this mistake fail silently and apply a',
                'wrong correction here the usage of J m-2 as unit is',
                'disallowed for now! Please check the input files and change',
                'the unit accordingly (e.g., using',
                '<cdo chunit,"J m**-2","J m**-2/day infile outfile>'])
            raise ValueError(errmsg)
        elif unit == 'J m**-2/day':
            da.data /= 24*60*60
            da.attrs['units'] = newunit
        else:
            logmsg = 'Unit {} not covered for {}'.format(unit, varn)
            raise ValueError(logmsg)

    # --- not covered ---
    else:
        logmsg = 'Variable {} not covered in standardize_units'.format(varn)
        logger.warning(logmsg)

    return da


def average_season(da, season, full_seasons_only=True):
    """
    An helper function for grouping and averaging season correctly.

    The xarray groupby('time.year').mean('time') routine does not account
    for seasons extending over multiple years (i.e., the winter season
    extending from December in year X to Jannuary and February in year X+1).
    This function fixes this but grouping consecutive months across different
    years.

    Parameters
    ----------
    da : xarray.DataArray
        Has to contain at least the time dimension.
    season : string {'JJA', 'SON', 'DJF', 'MAM', 'ANN'} or None
    full_seasons_only : bool, optional
        If True (default) use only full seasons, otherwise use all seasons.
        Note that using all seasons might lead to seasons consisting of only
        one month!

    Returns
    -------
    da_mean : same as input with time dimension averaged by season and renamed to year
        Note that for the winter season the new year dimension labels winters by the
        original year of January and February (i.e., the winter season 2000/2001 is
        labeled 2001!)
    """
    assert season in ['JJA', 'SON', 'DJF', 'MAM', 'ANN', None]
    if season != 'DJF':
        return da.groupby('time.year').mean('time', skipna=False)

    def get_season_label(time):
        """Label each season by the year of Jannyary & February"""
        if time.month == 12:
            return time.year+1
        return time.year

    if full_seasons_only:
        year_first = da.coords['time'].data[0].year
        year_last = da.coords['time'].data[-1].year
        logmsg = 'Dropping not-complete winter seasons: {}/{} and {}/{}'.format(
            year_first-1, year_first, year_last, year_last+1)
        logger.warning(logmsg)
        da = da.sel(time=slice('{}-12'.format(year_first), '{}-02'.format(year_last)))

    labels = [get_season_label(time) for time in da.coords['time'].data]
    groups = xr.DataArray(labels, dims=['time'], name='year')
    da_grouped = da.groupby(groups)

    # making sure everything is a expected
    if full_seasons_only:
        assert np.all([len(vv) == 3 for vv in da_grouped.groups.values()])
    else:
        months_per_group = [len(vv) for vv in da_grouped.groups.values()]
        assert months_per_group[0] == 2
        assert months_per_group[-1] == 1
        assert np.all([mm == 3 for mm in months_per_group[1:-1]])

    return da_grouped.mean('time')



def calculate_basic_diagnostic(infile, varn,
                               outfile=None,
                               id_=None,
                               time_period=None,
                               season=None,
                               time_aggregation=None,
                               mask_land_sea=False,
                               region='GLOBAL',
                               overwrite=False,
                               regrid=False,  # TODO, DELETE
                               idx_lats=None,
                               idx_lons=None):
    """
    Calculate a basic diagnostic from a given file.

    A basic diagnostic calculated from a input file by selecting a given
    region, time period, and season as well as applying a land-sea mask.
    Also, the time dimension is aggregated by different methods.

    Parameters
    ----------
    infile : str
        Full path of the input file. Must contain varn.
    varn : str
        The variable contained in infile.
    outfile : str, optional
        Full path of the output file. Path must exist.
    id_ : {'CMIP6', 'CMIP5', 'CMIP3', 'LE'}, optional
        A valid model ID
    time_period : tuple of two strings, optional
        Start and end of the time period. Both strings must be on of
        {"yyyy", "yyyy-mm", "yyyy-mm-dd"}.
    season : {'JJA', 'SON', 'DJF', 'MAM', 'ANN'}, optional
    time_aggregation : {'CLIM', 'STD', 'TREND', 'ANOM-GOBAL', 'ANOM-LOCAL'}, optional
        Type of time aggregation to use.
    mask_land_sea : {'sea', 'land', False}, optional
    region : list of strings or str, optional
        Each string must be a valid SREX region
    overwrite : bool, optional
        If True overwrite existing outfiles otherwise read and return them.
    regrid : bool, optional
        If True the file will be regridded by cdo.remapbil before opening.
    idx_lats : list of int, optional
    idx_lons : list of int, optional

    Returns
    -------
    diagnostic : xarray.DataArray
    """
    if not overwrite and outfile is not None and os.path.isfile(outfile):
        logger.debug('Diagnostic already exists & overwrite=False, skipping.')
        return xr.open_dataset(outfile, use_cftime=True)

    if id_ == 'CMIP6':  # need to concat historical file and delete 'height'
        scenario = infile.split('_')[-3]
        da = xr.open_dataset(infile, use_cftime=True)[varn]
        if scenario != 'historical':
            assert re.compile('[rcps]{3}[0-9]{3}$').match(scenario), 'not a scenario!'
            histfile = infile.replace(scenario, 'historical')
            da_hist = xr.open_dataset(histfile, use_cftime=True)[varn]
            da = xr.concat([da_hist, da], dim='time')
    else:
        da = xr.open_dataset(infile, use_cftime=True)[varn]

    try:
        da = da.drop_vars('height')
    except ValueError:
        pass

    da = standardize_units(da, varn)
    da = flip_antimeridian(da)
    assert np.all(da['lat'].data == np.arange(-88.75, 90., 2.5))
    assert np.all(da['lon'].data == np.arange(-178.75, 180., 2.5))

    if time_period is not None:
        da = da.sel(time=slice(str(time_period[0]), str(time_period[1])))

    # NOTE: CAMS-CSM1-0 is missing the last year!
    if str(time_period[1]) == '2100' and 'CAMS-CSM1-0' in infile:
        da = da.sel(time=slice(None, '2099'))

    if id_ in ['CMIP6', 'CMIP5', 'CMIP3', 'LE'] and np.any(np.isnan(da.data)):
        import ipdb; ipdb.set_trace()
        raise ValueError('Missing value in model detected!')

    if season in ['JJA', 'SON', 'DJF', 'MAM']:
        da = da.isel(time=da['time.season'] == season)
    elif season is None or season == 'ANN':
        pass
    else:
        raise NotImplementedError('season={}'.format(season))

    if isinstance(mask_land_sea, bool) and not mask_land_sea:
        pass
    elif mask_land_sea == 'sea':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sea_mask = regionmask.defined_regions.natural_earth.land_110.mask(da) == 0
        da = da.where(sea_mask)
    elif mask_land_sea == 'land':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            land_mask = np.isnan(regionmask.defined_regions.natural_earth.land_110.mask(da))
        da = da.where(land_mask)
    else:
        raise NotImplementedError

    if time_aggregation == 'ANOM-GLOBAL':
        da_mean = average_season(da, season)
        da_mean = da_mean.mean('year', skipna=False)
        da_mean = area_weighted_mean(da_mean)

    if region != 'GLOBAL':
        if (isinstance(region, str) and
            region not in regionmask.defined_regions.srex.abbrevs):
            # if region is not a SREX region read coordinate file
            regionfile = '{}.txt'.format(os.path.join(REGION_DIR, region))
            if not os.path.isfile(regionfile):
                raise ValueError(f'{regionfile} is not a valid regionfile')
            mask = np.loadtxt(regionfile)
            if mask.shape != (4, 2):
                errmsg = ' '.join([
                    f'Wrong file content for regionfile {regionfile}! Should',
                    'contain four lines with corners like: lon, lat'])
                raise ValueError(errmsg)
            lonmin, latmin = mask.min(axis=0)
            lonmax, latmax = mask.max(axis=0)
            if lonmax > 180 or lonmin < -180 or latmax > 90 or latmin < -90:
                raise ValueError(f'Wrong lat/lon value in {regionfile}')

            lats, lons = da['lat'].data, da['lon'].data
            lats = lats[(lats >= latmin) & (lats <= latmax)]
            lons = lons[(lons >= lonmin) & (lons <= lonmax)]
            da = da.sel(lat=lats, lon=lons)
        else:
            if isinstance(region, str):
                region = [region]
            masks = []
            keys = regionmask.defined_regions.srex.map_keys(region)
            for key in keys:
                masks.append(
                    regionmask.defined_regions.srex.mask(da) == key)
            mask = sum(masks) == 1
            da = da.where(mask, drop=True)

        if np.all(np.isnan(da.isel(time=0))):
            errmsg = 'All grid points masked! Wrong masking settings?'
            logger.error(errmsg)
            raise ValueError(errmsg)

    if idx_lats is not None and idx_lons is not None:
        da = da.isel(lat=idx_lats, lon=idx_lons)
        if np.all(np.isnan(da.isel(time=0))):
            # end program if only nan (i.e., ocean with mask)
            sys.exit(f'{idx_lats, idx_lons} contains only nan')

    attrs = da.attrs

    with warnings.catch_warnings():
        # suppress warnings on masked ocean grid cells
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')

        if time_aggregation == 'CLIM':
            # mean of seasonal (annual) means
            da = average_season(da, season)
            da = da.mean('year', skipna=False)
        elif time_aggregation == 'ANOM-LOCAL':
            da = average_season(da, season)
            da = da.mean('year', skipna=False)

            size = (~np.isnan(da)).sum()  # number of not NAN grid cells
            if size == 1:
                errmsg = ' '.join([
                    'ANOM-LOCAL is not possible for regions with only one',
                    'grid cell which is not NAN! Consider using ANOM-GLOBAL?'])
                logger.error(errmsg)
                raise ValueError(errmsg)
            elif size < 10:
                logmsg = ' '.join([
                    'ANOM-LOCAL is not recommended for regions with less than',
                    '10 gird cells which are not NAN! Consider using ANOM-GLOBAL?'])
                logger.warning(logmsg)
            da -= area_weighted_mean(da)
        elif time_aggregation == 'ANOM-GLOBAL':
            da = average_season(da, season)
            da = da.mean('year', skipna=False)
            da -= da_mean

        elif time_aggregation == 'STD':
            # standard deviation of de-trended seasonal (annual) means
            da = average_season(da, season)
            da = xr.apply_ufunc(detrend, da,
                                input_core_dims=[['year']],
                                output_core_dims=[['year']],
                                vectorize=True,
                                keep_attrs=True)
            da = da.std('year', skipna=False)
        elif time_aggregation in ['TREND', 'TREND-MEAN']:
            # trend of seasonal (annual) means
            da = average_season(da, season)
            da = xr.apply_ufunc(trend, da,
                                input_core_dims=[['year']],
                                output_core_dims=[[]],
                                vectorize=True,
                                keep_attrs=True)
            attrs['units'] = '{} year**-1'.format(attrs['units'])
        elif time_aggregation == 'CYC':
            # seasonal cycle over all years
            da = da.groupby('time.month').mean('time')
        elif time_aggregation is None or time_aggregation == 'CORR':
            pass
        else:
            NotImplementedError(f'time_aggregation={time_aggregation}')

    ds = da.to_dataset(name=varn)
    ds[varn].attrs = attrs
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

        if kwargs['idx_lats'] is None and kwargs['idx_lons'] is None:
            outfile = os.path.join(base_path, '_'.join([
                '{infile}_{time_period[0]}-{time_period[1]}_{season}',
                '{time_aggregation}_{region}_{masked}.nc']).format(
                    masked=(
                        kwargs['mask_land_sea'] + 'masked'
                        if not isinstance(kwargs['mask_land_sea'], bool) else 'unmasked'),
                    **kwargs))
        else:
            str_ = '_'.join(['-'.join(map(str, kwargs['idx_lats'])),
                             '-'.join(map(str, kwargs['idx_lons']))])
            outfile = os.path.join(base_path, '_'.join([
                '{infile}_{time_period[0]}-{time_period[1]}_{season}',
                '{time_aggregation}_{region}_{masked}_{str_}.nc']).format(
                    str_=str_,
                    masked=(
                        kwargs['mask_land_sea'] + 'masked'
                        if not isinstance(kwargs['mask_land_sea'], bool) else 'unmasked'),
                    **kwargs))
        return outfile

    if isinstance(diagn, str):  # basic diagnostic
        outfile = get_outfile(infile=infile, **kwargs)
        return calculate_basic_diagnostic(infile, diagn, outfile, **kwargs)
    elif isinstance(diagn, dict):  # derived diagnostic
        diagn = dict(diagn)  # leave original alone (.pop!)
        assert len(diagn.keys()) == 1
        key = list(diagn.keys())[0]
        varns = diagn.pop(key)  # basic variables
        diagn = key  # derived diagnostic

        if diagn == 'rnet':
            tmpfile = os.path.join(
                base_path, os.path.basename(infile).replace(varns[0], diagn))
            calculate_net_radiation(infile, varns, tmpfile, diagn)
            outfile = get_outfile(infile=tmpfile, **kwargs)
            return calculate_basic_diagnostic(tmpfile, diagn, outfile, **kwargs)
        elif kwargs['time_aggregation'] == 'CORR':
            assert len(varns) == 2, 'can only correlate two variables'
            assert varns[0] != varns[1], 'can not correlate same variables'
            outfile1 = get_outfile(infile=infile, **kwargs)
            ds1 = calculate_basic_diagnostic(infile, varns[0], outfile1, **kwargs)

            # !! '.../...Datasets...'.replace('tas', 'pr') -> '.../...Daprets...' !!
            # !! '.../processed... -> .../tasocessed... !! (for obs)
            path, fn = os.path.split(infile)
            fn = fn.replace(f'{varns[0]}_', f'{varns[1]}_')
            path = (path+'/').replace(f'/{varns[0]}/', f'/{varns[1]}/')
            infile2 = os.path.join(path, fn)
            outfile2 = get_outfile(infile=infile2, **kwargs)
            ds2 = calculate_basic_diagnostic(infile2, varns[1], outfile2, **kwargs)
            da = xr.apply_ufunc(correlation, ds1[varns[0]], ds2[varns[1]],
                                input_core_dims=[['time'], ['time']],
                                vectorize=True)
            outfile3 = outfile1.replace(f'/{varns[0]}_', f'/{diagn}_')
            ds3 = da.to_dataset(name=diagn)
            ds3[diagn].attrs = {'units': '1'}
            ds3.to_netcdf(outfile3)
            return ds3
