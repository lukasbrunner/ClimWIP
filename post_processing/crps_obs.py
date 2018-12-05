#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-11-30 13:56:54 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Calculate the CRPS of the target variable for the historical period
(if possible from the config file)

"""
import os
import argparse
import logging
import numpy as np
import xarray as xr
import properscoring as ps
from scipy.stats import norm

from utils_python import utils
from utils_python.xarray import area_weighted_mean
from utils_python.decorators import vectorize

from model_weighting.model_weighting import (
    set_up_filenames,
    calc_target,
)
from model_weighting.functions.diagnostics import calculate_diagnostic

logger = logging.getLogger(__name__)


def read_config():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='config', nargs='?', default='DEFAULT',
        help='Name of the configuration to use (optional).')
    parser.add_argument(
        '--filename', '-f', dest='filename', default='../configs/config.ini',
        help='Relative or absolute path/filename.ini of the config file.')
    args = parser.parse_args()
    cfg = utils.read_config(args.config, args.filename)
    utils.log_parser(cfg)

    cfg.target_startyear = cfg.predictor_startyears[0]
    cfg.target_endyear = cfg.predictor_endyears[0]
    cfg.target_startyear_ref = None
    cfg.target_endyear_ref = None
    cfg.target_agg = None

    return cfg


def read_weights(model_ensemble, cfg):
    filename = os.path.join(cfg.save_path, cfg.config + '.nc')
    ds = xr.open_dataset(filename)
    ds = ds.sel(model_ensemble=model_ensemble)
    return ds['weights']


@vectorize('(n),(n)->()')
def weighted_mean(data, weights):
    return np.average(data, weights=weights)


def main():
    """Call functions"""
    cfg = read_config()
    fn = set_up_filenames(cfg)
    targets = calc_target(fn, cfg)
    targets = area_weighted_mean(targets)  # models

    base_path = os.path.join(
        cfg.save_path, cfg.target_diagnostic, cfg.freq,
        'masked' if cfg.target_masko else 'unmasked')
    filename = os.path.join(
        cfg.obs_path, '{}_mon_{}_g025.nc'.format(
            cfg.target_diagnostic, cfg.obsdata))

    obs = calculate_diagnostic(
        filename, cfg.target_diagnostic, base_path,
        time_period=(
            cfg.target_startyear,
            cfg.target_endyear),
        season=cfg.target_season,
        time_aggregation=cfg.target_agg,
        mask_ocean=cfg.target_masko,
        region=cfg.region,
        overwrite=False,
        regrid=True,
    )[cfg.target_diagnostic]
    obs = area_weighted_mean(obs)

    weights = read_weights(targets['model_ensemble'].data, cfg)
    weights_time = np.tile(weights, (targets['time'].size, 1))

    targets = targets.transpose('time', 'model_ensemble')

    random_weights = np.random.rand(targets['model_ensemble'].size)
    random_weights = np.tile(random_weights, (targets['time'].size, 1))

    best_weights = weights
    weights[weights < np.sort(weights)[-10]] = 0.
    best_weights = np.tile(best_weights, (targets['time'].size, 1))

    baseline = ps.crps_ensemble(obs, targets).mean()
    weighted = ps.crps_ensemble(obs, targets, weights=weights_time).mean()
    random_w = ps.crps_ensemble(obs, targets, weights=random_weights).mean()
    best_w = ps.crps_ensemble(obs, targets, weights=best_weights).mean()

    print('CRPS vs baseline (unweighted model mean)')
    print(f'Weighted: {(baseline - weighted) / baseline:.2%}')
    print(f'Random weights: {(baseline - random_w) / baseline:.2%}')
    print(f'10 best: {(baseline - best_w) / baseline:.2%}')

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    utils.set_logger(level=logging.INFO)
    main()
