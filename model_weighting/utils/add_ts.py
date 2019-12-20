#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import xarray as xr

from utils_python.xarray import flip_antimeridian


def preprocess(ds):
    path = ds.encoding['source']
    model = os.path.basename(path).split('_')[2]
    ensemble = os.path.basename(path).split('_')[4]
    ds['model_ensemble'] = xr.DataArray(
        [f'{model}_{ensemble}_CMIP5'], dims='model_ensemble', name='model_ensemble')
    ds['time'] = xr.cftime_range(
        start=str(ds['time'].dt.year.data[0]), freq='M',
        periods=ds.dims['time'], calendar='noleap')
    return ds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='fn', type=str,
        help='Filename of a valid output file')
    args = parser.parse_args()

    ds = xr.open_dataset(args.fn)
    varn = ds.attrs['target']
    var = xr.open_mfdataset(
        ds['filename'].data, preprocess=preprocess,
        concat_dim='model_ensemble')[varn]
    var = flip_antimeridian(var)
    var = var.sel(lon=ds['lon'], lat=ds['lat'])
    var.load()
    ds[f'{varn}_ts'] = var

    ds.to_netcdf(args.fn.replace('.nc', '_ts.nc'))


if __name__ == '__main__':
    main()
