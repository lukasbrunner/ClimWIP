#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-11-12 21:52:34 lukas>

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
# import salem  # TODO: could use the roi function to cut regions
from cdo import Cdo
cdo = Cdo()

from utils_python.xarray import select_region, flip_antimeridian

logger = logging.getLogger(__name__)
REGION_DIR = '{}/../cdo_data/'.format(os.path.dirname(__file__))
MASK = 'land_sea_mask_regionsmask.nc'  # 'seamask_g025.nc'
unit_save = None


def calc_Rnet(infile, outname, variable, derived=True, workdir=None):
    """
    Procedure to calculate derived diagnostic net radiation.

    Parameters:
    - infile (string): Input filename incl. full path
    - outname (string): Output filename incl. full path but without extension
    - variable (string): one of the variables used in calculation present in infile
    optional:
    - derived (boolean): standard CMIP variable or derived from other variables?
    - workdir (string): temporary directory where intermediate files are
                        stored, default is in same location as script run
    Returns:
    - The name of the produced file (string)
    """
    if not derived:
        logger.error('This function is for derived variables, ' +
                     'something is wrong here.')
        sys.exit
    if not workdir:
        workdir = 'tmp_work'
    if (os.access(workdir, os.F_OK) == False):
        os.makedirs(workdir)

    variable1 = 'rlds'
    variable2 = 'rlus'
    variable3 = 'rsds'
    variable4 = 'rsus'

    # find files for other variables
    infile1 = infile.replace(variable, variable1)
    infile2 = infile.replace(variable, variable2)
    infile3 = infile.replace(variable, variable3)
    infile4 = infile.replace(variable, variable4)

    outname_rnet = outname.replace(variable, 'rnet')
    outfile = '%s.nc' %(outname_rnet)

    lwnet = cdo.sub(input = '%s %s' %(infile, infile2))
    swnet = cdo.sub(input = '%s %s' %(infile3, infile4))
    rnetfile = cdo.add(input = '%s %s' %(lwnet, swnet))
    cdo.chname('%s,rnet' %(variable), input = rnetfile,
               output = outfile)
    return outfile


def standardize_units(da, varn):
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


def calculate_basic_diagnostic(infile, varn, outfile,
                               time_period=None,
                               season=None,
                               time_aggregation=None,
                               mask_ocean=False,
                               region=None,
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

    if region is not None:
        if isinstance(region, str):
            region = [region]
        masks = []
        keys = regionmask.defined_regions.srex.map_keys(region)
        for key in keys:
            masks.append(
                regionmask.defined_regions.srex.mask(da) == key)
        mask = sum(masks) > 0
        da = da.where(mask)
        da = da.salem.subset(roi=da)  # TODO: test (but I think this is genius!)

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
        else:
            NotImplementedError('time_aggregation={}'.format(time_aggregation))

    ds = da.to_dataset(name=varn)
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
    outfile = os.path.join(
        base_path,
        '{infile}_{time_period[0]}-{time_period[1]}_{season}_{time_aggregation}_EUR_3SREX.nc'.format(  # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            infile=os.path.basename(infile).replace('.nc', ''), **kwargs))
    if isinstance(diagn, str):  # basic diagnostic
        return calculate_basic_diagnostic(infile, diagn, outfile, **kwargs)
    elif isinstance(diagn, dict):  # derived diagnostic
        assert len(diagn.keys()) == 1
        key = list(diagn.keys())[0]
        varns = diagn.pop(key)  # basic variables
        diagn = key  # derived diagnostic

        if diagn == 'rnet':
            derived_file = calc_Rnet(infile, outname, variable, derived=True)
            outname = outname.replace(variable, 'rnet')
            variable = 'rnet'
            logger.debug(outname)
        elif diagn == 'tasclt':
            calc_diag(infile, diagn, **kwargs)



def calc_CORR(infile,
              base_path,
              variable1,
              variable2,
              masko,
              syear,
              eyear,
              season,
              region,
              overwrite=False):
    """
    Procedure to calculate derived diagnostic correlation between
    two variables (e.g. temperature tas and cloud cover clt).

    Parameters:
    - infile (string): Input filename incl. full path
    - outname (string): Output filename incl. full path but without extension
    - variable1 (string): first variable in correlation
    - variable2 (string): second variable in correlation
    optional:
    - derived (boolean): standard CMIP variable or derived from other variables?
    - workdir (string): temporary directory where intermediate files are
                        stored, default is in same location as script run
    - masko (boolean): mask ocean or not?
    - landseamask (string): if masko = True we need the path to a land sea mask
    - syear (int): the start year if the original file is cut
    - eyear (int): the end year if the original file is cut
    - seasons (list of strings): a season JJA, MAM, DJF, SON or ANN for annual,
                                 if not given all of them calculated
    - region (string): a region name if the region should be cut
    - areadir (string): a path to the directory where the lat lon of the
                        regions are stored
    Returns:
    - Nothing, the produced file is outfile
    """

    # create template for output files
    outname = os.path.join(
        base_path,
        os.path.basename(infile).replace('clt', 'tasclt'))  # TODO, DEBUG !!! remove hardcoded stuff!
    outname = outname.replace('.nc', '')

    filename_global_kind = '{}_{}-{}_{}_CORR_GLOBAL.nc'.format(outname, syear, eyear, season)
    filename_region_kind = '{}_{}-{}_{}_CORR_{}.nc'.format(outname, syear, eyear, season, region)
    if not overwrite and (os.path.isfile(filename_global_kind) and
                          os.path.isfile(filename_region_kind)):
        logger.debug('Diagnostic already exists & overwrite=False, skipping.')
        if region == 'GLOBAL':
            return filename_global_kind
        return filename_region_kind

    base_path1 = base_path.replace('tasclt', variable1)
    os.makedirs(base_path1, exist_ok=True)
    outname = os.path.join(base_path1, os.path.basename(infile)).replace('.nc', '')

    filename_diag1 = calc_diag(infile, outname,
                               diagnostic=variable1,
                               masko=masko,
                               syear=syear,
                               eyear=eyear,
                               season=season,
                               region='GLOBAL',
                               overwrite=overwrite,
                               kind=None)

    base_path2 = base_path.replace('tasclt', variable2)
    os.makedirs(base_path2, exist_ok=True)
    infile2 = infile.replace(variable1, variable2)
    outname = os.path.join(base_path2, os.path.basename(infile2)).replace('.nc', '')
    filename_diag2 = calc_diag(infile2, outname,
                               diagnostic=variable2,
                               masko=masko,
                               syear=syear,
                               eyear=eyear,
                               season=season,
                               region='GLOBAL',
                               overwrite=overwrite,
                               kind=None)

    with TemporaryDirectory(dir='/net/h2o/climphys/tmp') as tmpdir:
        tmpfile = os.path.join(tmpdir, 'temp.nc')
        tmpfile2 = os.path.join(tmpdir, 'temp2.nc')

        cdo.timcor(input='{} {}'.format(filename_diag1, filename_diag2),
                   output=tmpfile)
        cdo.chname('%s,%s%s' %(variable1, variable2, variable1),
                   input=tmpfile, output=tmpfile2)
        cdo.setattribute('%s%s@units=-' %(variable2, variable1),
                         input=tmpfile2, output=tmpfile)
        cdo.setvrange('-1,1', input=tmpfile, output=filename_global_kind)

    if region != 'GLOBAL':
        mask = np.loadtxt('%s/%s.txt' %(REGION_DIR, region))
        lonmax = np.max(mask[:, 0])
        lonmin = np.min(mask[:, 0])
        latmax = np.max(mask[:, 1])
        latmin = np.min(mask[:, 1])

        cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                         input=filename_global_kind,
                         output=filename_region_kind)

    if region == 'GLOBAL':
        return filename_global_kind
    return filename_region_kind
