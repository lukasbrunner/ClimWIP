#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-03-14 10:30:57 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Calculate the CRPS of the target variable for the future period
based on a perfect model test.

"""
import os
import warnings
import numpy as np
import xarray as xr
from properscoring import crps_ensemble

from utils_python.xarray import area_weighted_mean
from utils_python.decorators import vectorize


PATH = '/net/h2o/climphys1/lukbrunn/Data/ModelWeighting/Paper'
FILES = ['eu_perfect.nc', 'neu_perfect.nc', 'ceu_perfect.nc', 'med_perfect.nc']
VARN = 'tas'


@vectorize('(m),(m)->()', excluded=[2, 3])
def crps_data(data, weights, how='default', nr=10):
    """calculate the CRPS"""
    # get the index of the perfect model
    idx = np.where(np.isnan(weights))[0][0]

    # extract and delete the perfect model from the data
    data_ref = data[idx]
    data = np.delete(data, idx)
    weights = np.delete(weights, idx)

    if how == 'default':
        pass
    elif how == 'shuffle_weights':
        np.random.shuffle(weights)
    elif how == 'best_x':
        idx = np.where(weights < np.sort(weights)[-nr])[0]
        weights[idx] = 0.
    elif how == 'best_x_equal':
        idx = np.where(weights < np.sort(weights)[-nr])[0]
        weights[:] = 1.
        weights[idx] = 0.
    elif how == 'random_x':
        idx = np.random.choice(np.arange(len(weights)), len(weights)-nr, replace=False)
        weights[idx] = 0.
    elif how == 'random_x_equal':
        idx = np.random.choice(np.arange(len(weights)), len(weights)-nr, replace=False)
        weights[:] = 1.
        weights[idx] = 0.

    baseline = crps_ensemble(data_ref, data)
    weighted = crps_ensemble(data_ref, data, weights=weights)
    return (baseline - weighted) / baseline


def crps_xarray(ds, varn):
    """xarray calls"""
    methods = ['default', 'best_x', 'best_x_equal', 'shuffle_weights',
               'random_x', 'random_x_equal']
    skills = {}
    for method in methods:
        skills[method] = xr.apply_ufunc(
            crps_data, ds[varn], ds['weights'],
            input_core_dims=[['model_ensemble'], ['model_ensemble']],
            kwargs={'how': method})
    return skills


def main():
    """load files and call functions"""
    print('config              : metric              : median  | mean    (90% range)')
    print(''.join(['-']*70))

    for ff in FILES:
        ds = xr.open_dataset(os.path.join(PATH, ff))
        ds = area_weighted_mean(ds, suppress_warning=True)

        skills = crps_xarray(ds, VARN)

        for key, skill in skills.items():
            warnings.simplefilter('ignore')
            median = skill.median('perfect_model_ensemble').data
            mean = skill.mean('perfect_model_ensemble').data
            p05, p95 = skill.quantile((.05, .95), 'perfect_model_ensemble').data
            print(f'{ff:<20}: {key:<20}: {median:>+7.2%} | {mean:>+7.2%} ({p05:>+8.2%} to {p95:>+8.2%})')

        print()


if __name__ == "__main__":
    main()
