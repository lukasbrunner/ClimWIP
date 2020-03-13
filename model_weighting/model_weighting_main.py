#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2020 Lukas Brunner, ETH Zurich

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

Abstract
    --------
Main script of the model weighting scheme described by Brunner et al. (2019)
Lorenz et al. (2018) and Knutti et al. (2017). If you use this code please note
the license and consider citing the papers below.

References
----------
Knutti, R., J. Sedláček, B. M. Sanderson, R. Lorenz, E. M. Fischer, and
V. Eyring (2017), A climate model projection weighting scheme accounting
for performance and interdependence, Geophys. Res. Lett., 44, 1909–1918,
 doi:10.1002/2016GL072012.

Lorenz, R., Herger, N., Sedláček, J., Eyring, V., Fischer, E. M., and
Knutti, R. (2018). Prospects and caveats of weighting climate models for
summer maximum temperature projections over North America. Journal of
Geophysical Research: Atmospheres, 123, 4509–4526. doi:10.1029/2017JD027992.

Brunner, L., R. Lorenz, M. Zumwald, R. Knutti (2019): Quantifying uncertainty
in European climate projections using combined performance-independence weighting.
Eniron. Res. Lett., https://doi.org/10.1088/1748-9326/ab492f
"""
import os
import logging
import argparse
import warnings
import numpy as np
import xarray as xr
from natsort import natsorted

from core.get_filenames import get_filenames, select_variants
from core.diagnostics import calculate_diagnostic
from core.perfect_model_test import perfect_model_test
from core.read_config import read_config
from core.process_variants import (
    process_variants,
    process_variants_target,
    independence_sigma_from_variants,
    expand_variants,
)
from core.weights import (
    calculate_weights_sigmas,
    calculate_weights,
    independence_sigma,
)
from core import utils
from core.utils_xarray import (
    add_revision,
    area_weighted_mean,
    weighted_distance_matrix,
    distance_matrix,
    distance_uncertainty
)
from core.plots import (
    plot_rmse,
    plot_maps,
    plot_fraction_matrix,
    plot_weights,
)

logger = logging.getLogger(__name__)

DERIVED = {
    'tashuss': ('huss', 'tas'),
    'tasclt': ('clt', 'tas'),
    'taspr': ('pr', 'tas'),
    'rnet': ('rlds', 'rlus', 'rsds', 'rsus'),
    'ef': ('hfls', 'hfss'),
    'dtr': ('tasmax', 'tasmin'),
}
REGRID_OBS = [
    'ERA-Interim']


def read_args():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='config', nargs='?', default='DEFAULT',
        help='Name of the configuration to use (optional).')
    parser.add_argument(
        '--filename', '-f', dest='filename', default='configs/config.ini',
        help='Relative or absolute path/filename.ini of the config file.')
    parser.add_argument(
        '--logging-level', '-log-level', dest='log_level', default=20,
        type=str, choices=['error', 'warning', 'info', 'debug'],
        help='Set logging level')
    parser.add_argument(
        '--logging-file', '-log-file', dest='log_file', default=None,
        type=str, help='Redirect logging output to given file')
    return parser.parse_args()


def calc_target(filenames, cfg):
    """
    Calculates the target variable for each model.

    Parameters
    ----------
    filenames : dictionary
        See get_filenames() docstring for more information.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    targets : xarray.DataArray, shape (L, M, N)
        DataArray of targets depending on models, lat, lon
    targets_clim : xarray.DataArray, shape (L, M, N)
        DataArray of target climatologies. If no reference period is given,
        this will be None.
    """
    # build and create path
    base_path = os.path.join(cfg.save_path, cfg.target_diagnostic)
    os.makedirs(base_path, exist_ok=True)

    targets = []
    clims = []
    for model_ensemble, filename in filenames.items():
        with utils.LogTime(model_ensemble, level='debug'):
            target = calculate_diagnostic(
                infile=filename,
                diagn=cfg.target_diagnostic,
                id_=model_ensemble.split('_')[2],
                base_path=base_path,
                time_period=(
                    cfg.target_startyear,
                    cfg.target_endyear),
                season=cfg.target_season,
                time_aggregation=cfg.target_agg,
                mask_land_sea=cfg.target_mask,
                region=cfg.target_region,
                overwrite=cfg.overwrite,
                idx_lats=cfg.idx_lats,
                idx_lons=cfg.idx_lons,
            )

            # calculate change rather than absolute value
            if cfg.target_startyear_ref is not None:
                target_hist = calculate_diagnostic(
                    infile=filename,
                    id_=model_ensemble.split('_')[2],
                    diagn=cfg.target_diagnostic,
                    base_path=base_path,
                    time_period=(
                        cfg.target_startyear_ref,
                        cfg.target_endyear_ref),
                    season=cfg.target_season,
                    time_aggregation=cfg.target_agg,
                    mask_land_sea=cfg.target_mask,
                    region=cfg.target_region,
                    overwrite=cfg.overwrite,
                    idx_lats=cfg.idx_lats,
                    idx_lons=cfg.idx_lons,
                )
                target[cfg.target_diagnostic] -= target_hist[cfg.target_diagnostic]
                target_hist['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')

                clims.append(target_hist)

        target['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        targets.append(target)
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

    return (xr.concat(targets, dim='model_ensemble')[cfg.target_diagnostic],
            xr.concat(clims, dim='model_ensemble')[cfg.target_diagnostic]
            if cfg.target_startyear_ref is not None else None)


def calc_performance(filenames, cfg):
    """
    Calculate the performance predictor diagnostics for each model.

    For each predictor calculate the corresponding diagnostic, i.e.,
    the area weighted RMSE between each model and the observations.

    Parameters
    ----------
    filenames : nested dictionary
        See get_filenames() docstring for more information.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    differences : xarray.DataArray, shape (N, M)
        A data array with dimensions (number of diagnostics, number of models).
    """
    # for each file in filenames calculate all diagnostics for each time period
    diffs = []
    for idx, diagn in enumerate(cfg.performance_diagnostics):
        logger.info(f'Calculate performance diagnostic {diagn}{cfg.performance_aggs[idx]}...')

        base_path = os.path.join(cfg.save_path, diagn)
        os.makedirs(base_path, exist_ok=True)

        # if its a derived diagnostic: get first basic variable to get one of
        # the filenames (the others will be created by replacement)
        varn = [*diagn.values()][0][0] if isinstance(diagn, dict) else diagn
        diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

        diagnostics = []
        for model_ensemble, filename in filenames[varn].items():
            with utils.LogTime(model_ensemble, level='debug'):

                diagnostic = calculate_diagnostic(
                    infile=filename,
                    id_=model_ensemble.split('_')[2],
                    diagn=diagn,
                    base_path=base_path,
                    time_period=(
                        cfg.performance_startyears[idx],
                        cfg.performance_endyears[idx]),
                    season=cfg.performance_seasons[idx],
                    time_aggregation=cfg.performance_aggs[idx],
                    mask_land_sea=cfg.performance_masks[idx],
                    region=cfg.performance_regions[idx],
                    overwrite=cfg.overwrite,
                    idx_lats=cfg.idx_lats,
                    idx_lons=cfg.idx_lons,
                )

            diagnostic['model_ensemble'] = xr.DataArray(
                [model_ensemble], dims='model_ensemble')
            diagnostics.append(diagnostic)
            logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

        diagnostics = xr.concat(diagnostics, dim='model_ensemble')

        logger.debug('Read observations & calculate model quality...')
        obs_list = []
        for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
            filename = os.path.join(obs_path, f'{varn}_mon_{obs_id}_g025.nc')

            with utils.LogTime(f'Calculate diagnostic for {obs_id}', level='debug'):
                obs = calculate_diagnostic(
                    infile=filename,
                    diagn=diagn,
                    base_path=base_path,
                    time_period=(
                        cfg.performance_startyears[idx],
                        cfg.performance_endyears[idx]),
                    season=cfg.performance_seasons[idx],
                    time_aggregation=cfg.performance_aggs[idx],
                    mask_land_sea=cfg.performance_masks[idx],
                    region=cfg.performance_regions[idx],
                    overwrite=cfg.overwrite,
                    regrid=obs_id in REGRID_OBS,
                    idx_lats=cfg.idx_lats,
                    idx_lons=cfg.idx_lons,
                )
                obs_list.append(obs)

        obs = xr.concat(obs_list, dim='dataset_dim')

        # NOTE: calculate differences based on global mean properties
        if cfg.performance_aggs[idx] in ['CLIM-MEAN', 'TREND-MEAN']:
            if cfg.obs_uncertainty == 'range':
                raise NotImplementedError
            diagnostics[diagn_key] = area_weighted_mean(diagnostics[diagn_key])
            obs = area_weighted_mean(obs)
        # ---

        if cfg.obs_uncertainty == 'range':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                obs_min = obs.min('dataset_dim', skipna=False)
                obs_max = obs.max('dataset_dim', skipna=False)
                diff = xr.apply_ufunc(
                    distance_uncertainty, diagnostics[diagn_key],
                    obs_min[diagn_key], obs_max[diagn_key],
                    input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
                    output_core_dims=[['lat', 'lon']],
                    vectorize=True)
        elif cfg.obs_uncertainty == 'center':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                obs_min = obs.min('dataset_dim', skipna=False)
                obs_max = obs.max('dataset_dim', skipna=False)
            diff = diagnostics[diagn_key] - .5*(obs_min[diagn_key] + obs_max[diagn_key])
        elif cfg.obs_uncertainty == 'mean':
            diff = diagnostics[diagn_key] - obs.mean('dataset_dim', skipna=False)[diagn_key]
        elif cfg.obs_uncertainty == 'median':
            diff = diagnostics[diagn_key] - obs.median('dataset_dim', skipna=False)[diagn_key]
        elif cfg.obs_uncertainty is None:
            # obs_min and obs_max are the same for this case
            diff = diagnostics[diagn_key] - obs[diagn_key].squeeze()
        else:
            raise NotImplementedError

        # --- plot map of each difference ---
        # takes a long time -> only commend in if needed
        # if cfg.plot:
        #     with utils.LogTime('Plotting maps', level='info'):
        #         plot_maps(diff, idx, cfg)
        # ---------------------------------------

        if cfg.performance_aggs[idx] in ['CLIM-MEAN', 'TREND-MEAN']:
            diff = np.abs(diff)
        else:
            diff = np.sqrt(area_weighted_mean(diff**2))

        if cfg.performance_aggs[idx] == 'CYC':
            diff = diff.mean('month')

        diff = diff.expand_dims({'diagnostic': [idx]})
        logger.debug('Read observations & calculate model quality... DONE')

        diffs.append(diff)
        logger.info(f'Calculate performance diagnostic {diagn_key}{cfg.performance_aggs[idx]}... DONE')

    return xr.concat(diffs, dim='diagnostic')


def calc_independence(filenames, cfg):
    """
    TODO
    Calculates the independence predictor diagnostics for each model.

    For each predictor calculate the corresponding diagnostic, i.e.,
    the area weighted RMSE between each model pair.

    Parameters
    ----------
    filenames : nested dictionary
        See get_filenames() docstring for more information.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    differences : xarray.DataArray, shape (N, M, M)
        A data array with dimensions (number of diagnostics, number of models,
        number of models).
    """
    # for each file in filenames calculate all diagnostics for each time period
    diffs = []
    for idx, diagn in enumerate(cfg.independence_diagnostics):
        logger.info(f'Calculate independence diagnostic {diagn}{cfg.independence_aggs[idx]}...')

        base_path = os.path.join(cfg.save_path, diagn)
        os.makedirs(base_path, exist_ok=True)

        # if its a derived diagnostic: get first basic variable to get one of
        # the filenames (the others will be created by replacement)
        varn = [*diagn.values()][0][0] if isinstance(diagn, dict) else diagn
        diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

        diagnostics = []
        for model_ensemble, filename in filenames[varn].items():
            with utils.LogTime(model_ensemble, level='debug'):

                diagnostic = calculate_diagnostic(
                    infile=filename,
                    id_=model_ensemble.split('_')[2],
                    diagn=diagn,
                    base_path=base_path,
                    time_period=(
                        cfg.independence_startyears[idx],
                        cfg.independence_endyears[idx]),
                    season=cfg.independence_seasons[idx],
                    time_aggregation=cfg.independence_aggs[idx],
                    mask_land_sea=cfg.independence_masks[idx],
                    region=cfg.independence_regions[idx],
                    overwrite=cfg.overwrite,
                    idx_lats=cfg.idx_lats,
                    idx_lons=cfg.idx_lons,
                )

            diagnostic['model_ensemble'] = xr.DataArray(
                [model_ensemble], dims='model_ensemble')
            diagnostics.append(diagnostic)
            logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

        diagnostics = xr.concat(diagnostics, dim='model_ensemble')
        logger.debug('Calculate model independence matrix...')

        if cfg.performance_aggs[idx] in ['CLIM-MEAN', 'TREND-MEAN']:
            diff = xr.apply_ufunc(
                distance_matrix, area_weighted_mean(diagnostics[diagn_key]),
                input_core_dims=[['model_ensemble']],
                output_core_dims=[['perfect_model_ensemble', 'model_ensemble']]
            )
        else:
            diff = xr.apply_ufunc(
                weighted_distance_matrix, diagnostics[diagn_key],
                input_core_dims=[['model_ensemble', 'lat', 'lon']],
                output_core_dims=[['perfect_model_ensemble', 'model_ensemble']],
                kwargs={'lat': diagnostics['lat'].data},  # NOTE: comment out for unweighted
                vectorize=True,
            )
        # fill newly defined dimension
        diff['perfect_model_ensemble'] = diff['model_ensemble'].data

        if cfg.independence_aggs[idx] == 'CYC':
            diff = diff.mean('month')

        diff.name = 'data'
        diff = diff.expand_dims({'diagnostic': [idx]})
        diffs.append(diff)

        logger.debug('Calculate independence matrix... DONE')

        logger.info(f'Calculate independence diagnostic {diagn}{cfg.independence_aggs[idx]}...DONE')
    return xr.concat(diffs, dim='diagnostic')


def _normalize(data, normalize_by):
    """Apply different normalization schemes to the right dimensions"""
    normalize_by = normalize_by[0]
    try:
        normalize_by = float(normalize_by)
        assert normalize_by > np.nanmin(data) / 10.
        assert normalize_by < np.nanmax(data) * 10
        normalizer = normalize_by
    except ValueError:
        if normalize_by.lower() == 'center':
            normalizer = .5*(np.nanmin(data) + np.nanmax(data))
        elif normalize_by.lower() == 'median':
            normalizer = np.nanmedian(data)
        elif normalize_by.lower() == 'mean':
            normalizer = np.nanmean(data)
        else:
            raise ValueError

    # # NOTE: optionally write out the normalizer for later use
    # with open('../data/normalizer_tasANOM-GLOBAL.tex', 'a') as ff:
    #     ff.write(f'{normalizer}\n')

    return data / normalizer


def calc_deltas(performance_diagnostics, independence_diagnostics, cfg):
    """
    Normalize and average diagnostics for performance and independence.

    Parameters
    ----------
    performance_diagnostics : xarray.DataArray, shape(N, M)
        See calc_performance()
    independence_diagnostics : xarray.DataArray, shape(N, M, M)
        See calc_independence()
    cfg : config object

    Returns
    -------
    delta_p : xarray.DataArray, shape(M,) or None
    delta_i : xarray.DataArray, shape(M, M)
    """
    # make this to DataArray
    # NOTE: I belief the 'temp' dimension is necessary to pass to xarray.apply_ufunc
    # as core dimension. I can not pass no dimension because this will be interpreted
    # as 'use all dimensions'.
    normalize_by = xr.DataArray([cfg.independence_normalizers], dims=('temp', 'diagnostic'),
                                coords={'diagnostic': independence_diagnostics['diagnostic']})
    independence_diagnostics = xr.apply_ufunc(
        _normalize, independence_diagnostics, normalize_by,
        input_core_dims=[['model_ensemble', 'perfect_model_ensemble'], ['temp']],
        # NOTE: we want the perfect model dimension to be the first one for subsequent uses!
        output_core_dims=[['perfect_model_ensemble', 'model_ensemble']],
        vectorize=True)
    delta_i, sigma_i, independence_diagnostics = process_variants(independence_diagnostics, cfg)
    delta_i.name = 'delta_i'

    if cfg.plot:
        max_ = np.max([np.nanpercentile(dd, 95) for dd in independence_diagnostics])
        min_ = np.min([np.nanpercentile(dd, 5) for dd in independence_diagnostics])
        plot_rmse(delta_i, 'mean', cfg, 'independence')
        for idx, diag in enumerate(independence_diagnostics):
            plot_rmse(diag, idx, cfg, 'independence', min_, max_)

    if performance_diagnostics is None:  # model-model distances only
        return None, delta_i, None

    normalize_by = xr.DataArray([cfg.performance_normalizers], dims=('temp', 'diagnostic'),
                                coords={'diagnostic': performance_diagnostics['diagnostic']})
    performance_diagnostics = xr.apply_ufunc(
        _normalize, performance_diagnostics, normalize_by,
        input_core_dims=[['model_ensemble'], ['temp']],
        output_core_dims=[['model_ensemble']],
        vectorize=True)
    delta_q, _, performance_diagnostics = process_variants(performance_diagnostics, cfg)
    delta_q.name = 'delta_q'

    if cfg.plot:
        max_ = np.max([np.percentile(dd, 95) for dd in performance_diagnostics])
        min_ = np.min([np.percentile(dd, 5) for dd in performance_diagnostics])
        plot_rmse(delta_q, 'mean', cfg, 'performance')
        for idx, diag in enumerate(performance_diagnostics):
            plot_rmse(diag, idx, cfg, 'performance', min_, max_)

    if cfg.variants_independence:
        logger.info('sigma_i estimated from ensemble variants: {:.4f}'.format(sigma_i[0]))

    return delta_q, delta_i, sigma_i


def calc_sigmas(targets, delta_i, sigma_i_variants, cfg, n_sigmas=50):
    """
    Perform a perfect model test to estimate the optimal shape parameters.

    Perform a perfect model test to estimate the optimal shape parameters for
    model quality and independence. See, e.g., Knutti et al., 2017.

    Parameters
    ----------
    targets : xarray.DataArray, shape (L, M, N)
        Array of targets. Should depend on models, lat, lon
    delta_i : xarray.DataArray, shape (L, L)
        A full distance matrix.
    unique_models : list of strings
        A list with model identifiers to select only one member per model.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    sigma_q : float
        Optimal shape parameter for quality weighing
    sigma_i : float
        Optimal shape parameter for independence weighting
    """
    if cfg.sigma_i is not None and cfg.sigma_q is not None:
        logger.info('Using user sigmas: q={}, i={}'.format(cfg.sigma_q, cfg.sigma_i))
        return cfg.sigma_q, cfg.sigma_i

    sigma_base = np.nanmean(delta_i)  # an estimated sigma to start

    # a large value means all models have equal quality -> we want this as small as possible
    if isinstance(cfg.sigma_q, (int, float)):
        # if the user has given one of the sigmas use it
        # NOTE: if sigma is -99 the corresponding weights will be set to 1 by convention
        sigmas_q = np.array([cfg.sigma_q])
    else:
        # otherwise allow a range
        sigmas_q = np.linspace(.2*sigma_base, 2*sigma_base, n_sigmas)
    # a large value means all models depend on each other, a small value means all models
    # are independent -> we want this ~delta_i
    if isinstance(cfg.sigma_i, (int, float)):
        sigmas_i = np.array([cfg.sigma_i])
    elif sigma_i_variants is not None:
        sigmas_i = sigma_i_variants
    else:
        sigmas_i = np.linspace(.2*sigma_base, 2*sigma_base, n_sigmas)

    if len(sigmas_q) == 1 and len(sigmas_i) == 1:
        logger.info('Using sigmas: q={}, i={}'.format(sigmas_q[0], sigmas_i[0]))
        return sigmas_q[0], sigmas_i[0]

    targets_mean = area_weighted_mean(targets)
    targets_mean = process_variants_target(targets_mean, cfg)

    if cfg.variants_combine:
        # in this case they only include one variant per model already
        targets_mean_1ens = targets_mean
        delta_i_1ens = delta_i
    else:
        # NOTE: if we take the mean first it does not matter how we sort them, but if
        # we don't it does! Particularly 'random' might lead to different results
        # every time.
        _, models_1ens = select_variants(targets['model_ensemble'].data, 1, 'natsorted')

        if len(models_1ens) > 1:
            # for the perfect model test we only use one member per model!
            targets_mean_1ens = targets_mean.sel(model_ensemble=models_1ens).data
            delta_i_1ens = delta_i.sel(model_ensemble=models_1ens,
                                       perfect_model_ensemble=models_1ens).data
        else:  # if we run this with only one SMILE use all variants
            targets_mean_1ens = targets_mean
            delta_i_1ens = delta_i

    # use the initial-condition members to estimate sigma_i
    if cfg.variants_independence and cfg.variants_combine is None:
        # old way if variants are not combined
        sigmas_i = independence_sigma(delta_i, sigmas_i)

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    # ratio of perfect models inside their respective weighted percentiles
    # for each sigma combination
    inside_ratio = perfect_model_test(
        targets_mean_1ens, weights_sigmas,
        perc_lower=cfg.percentiles[0],
        perc_upper=cfg.percentiles[1])

    force_inside_ratio = isinstance(cfg.inside_ratio, str)  # then it is 'force'
    if force_inside_ratio:
        cfg.inside_ratio = cfg.percentiles[1] - cfg.percentiles[0]
    inside_ok = inside_ratio >= cfg.inside_ratio

    if not np.any(inside_ok):
        logmsg = f'Perfect model test failed ({inside_ratio.max():.4f} < {cfg.inside_ratio:.4f})!'
        if force_inside_ratio:
            # adjust inside_ratio to force a result (probably not recommended?)
            inside_ok = inside_ratio >= np.max(inside_ratio)
            logmsg += ' force=True: Setting inside_ratio to max: {}'.format(
                np.max(inside_ratio))
            logger.warning(logmsg)
        else:
            raise ValueError(logmsg)

    # find the element with the smallest sum i+j which is True
    index_sum = 9999
    idx_q_min = None
    for idx_q, qq in enumerate(inside_ok):
        if qq.sum() == 0:
            continue  # no fitting element
        elif idx_q >= index_sum:
            break  # no further optimization possible
        idx_i = np.where(qq)[0][0]
        if idx_i + idx_q < index_sum:
            index_sum = idx_i + idx_q
            idx_i_min, idx_q_min = idx_i, idx_q

    logger.info('sigma_q: {:.4f}; sigma_i: {:.4f}'.format(
        sigmas_q[idx_q_min], sigmas_i[idx_i_min]))

    if cfg.plot:
        plot_fraction_matrix(
            sigmas_i, sigmas_q, inside_ratio, cfg, (idx_i_min, idx_q_min),
            'Fraction of models within {} to {} percentile'.format(
                *cfg.percentiles))

    return sigmas_q[idx_q_min], sigmas_i[idx_i_min]


def calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg):
    """
    Calculate the weights for given set of parameters.

    Calculate the weights for given model qualities (delta_q), model
    independence (detal_i), and shape parameters sigma_q and sigma_i.

    Parameters
    ----------
    delta_q : xarray.DataArray, shape (N,) or (N, N)
        Array of distances from each model to the observations.
    delta_i : xarray.DataArray, shape (N, N)
        Array of distances from each model to each other model.
    sigma_q : float
        Float giving the quality weighting shape parameter.
    sigma_i : float
        Float giving the independence weighting shape parameter.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    weights : xarray.Dataset
        A Dataset containing several DataArrays.
    """
    # make sure the perfect model dimension is still the first one!
    delta_i = delta_i.transpose('perfect_model_ensemble', 'model_ensemble')

    if cfg.obs_id is not None:
        numerator, denominator = calculate_weights(delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= weights.sum()
        dims = 'model_ensemble'
    else:
        # NOTE: without observations delta_q does not exist but we use the
        # perfect model setting which is stored in delta_i and create weights
        # for each case!
        delta_q = delta_i
        delta_q.name = 'delta_q'
        calculate_weights_matrix = np.vectorize(
            calculate_weights, signature='(n)->(n),(n)', excluded=[1, 2, 3, 4])
        numerator, denominator = calculate_weights_matrix(
            delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= np.nansum(weights, axis=-1)
        dims = ('perfect_model_ensemble', 'model_ensemble')

    # create this just to fill
    ds = delta_q.to_dataset().copy()
    ds = ds.rename({'delta_q': 'weights'})
    ds['weights'].data = weights
    ds['weights_q'] = xr.DataArray(numerator, dims=dims)
    ds['weights_i'] = xr.DataArray(denominator, dims=dims)
    ds['delta_q'] = delta_q
    ds['delta_i'] = delta_i
    ds['sigma_q'] = xr.DataArray(sigma_q)
    ds['sigma_i'] = xr.DataArray(sigma_i)

    # add some metadata
    ds['model_ensemble'].attrs = {
        'units': '1',
        'long_name': 'Unique Model Identifier',
        'description': ' '.join([
            'Underscore-separated model identifyer:',
            'model_ensemble_project']),
    }
    ds['perfect_model_ensemble'].attrs = {
        'units': '1',
        'long_name': 'Unique Perfect Model Identifier',
        'description': ' '.join([
            'Underscore-separated perfect model (c.f. perfect model test) identifyer:',
            'model_ensemble_project'])
    }
    ds['weights'].attrs = {
        'units': '1',
        'long_name': 'Normalized Model Weights',
        'description': '(weights_q/weights_i) / sum(weights_q/weights_i)'
    }
    ds['weights_q'].attrs = {
        'units': '1',
        'long_name': 'Quality Weights (not Normalized)',
    }
    ds['weights_i'].attrs = {
        'units': '1',
        'long_name': 'Independence Weights (not Normalized)',
        'description': 'Higher values mean more dependence!',
    }
    ds['delta_q'].attrs = {
        'units': '1',
        'long_name': 'Observational Distance Metric',
    }
    ds['delta_i'].attrs = {
        'units': '1',
        'long_name': 'Model Distance Metric',
    }
    ds['sigma_q'].attrs = {
        'units': '1',
        'long_name': 'Observational Distance Shape Parameter',
    }
    ds['sigma_i'].attrs = {
        'units': '1',
        'long_name': 'Model Distance Shape Parameter',
    }

    if cfg.plot and cfg.obs_id:
        plot_weights(ds, cfg)
        plot_weights(ds, cfg, sort_by='weight')
        plot_weights(ds, cfg, sort_by='performance')
        plot_weights(ds, cfg, sort_by='independence')

    return ds


def save_data(ds, targets, clim, filenames, cfg):
    """Save the given Dataset to a file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save
    targets : None or xarray.DataArray
        Target DataArray to add to ds
    clim : None or xarray.DataArray
        Target climatology to add to ds
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    None
    """
    if targets is not None:
        targets = targets.sel(model_ensemble=natsorted(targets['model_ensemble'].data))
        if cfg.variants_combine:  # targets still has all variants!
            ds = expand_variants(ds, targets['model_ensemble'].data)
            targets_mean = process_variants_target(targets, cfg)
            targets_mean = targets_mean.rename({'model_ensemble': 'model'})
            ds[f'{cfg.target_diagnostic}_mean'] = targets_mean

        # save additional variables for convenience
        ds[cfg.target_diagnostic] = targets
        ds['filename'] = xr.DataArray(
            [*filenames[cfg.target_diagnostic].values()],
            coords={'model_ensemble': [*filenames[cfg.target_diagnostic].keys()]},
            dims='model_ensemble',
            attrs={
                'units': '1',
                'long_name': 'Full Path and Filename',
            })

        ds.attrs['target'] = cfg.target_diagnostic
        ds.attrs['region'] = cfg.target_region

    if clim is not None:
        clim = clim.sel(model_ensemble=natsorted(clim['model_ensemble'].data))
        if cfg.variants_combine:
            clim_mean = process_variants_target(clim, cfg)
            ds[f'{cfg.target_diagnostic}_clim_mean'] = clim_mean
        ds[f'{cfg.target_diagnostic}_clim'] = clim

    # add some metadata
    ds.attrs.update({
        'config': cfg.config,
        'config_path': cfg.config_path,
        'reference': ' '.join([
            'Brunner et al. (2019): Quantifying uncertainty in European',
            'climate projections using combined performance-independence',
            'weighting. Eniron. Res. Lett.',
            'https://doi.org/10.1088/1748-9326/ab492f.']),
        'git_hash': utils.get_git_info(),
    })
    add_revision(ds)

    filename = os.path.join(cfg.save_path, f'{cfg.config}.nc')
    ds.to_netcdf(filename)
    logger.info('Saved file: {}'.format(filename))


def main(args):
    """Call functions"""
    log = utils.LogTime()

    log.start('main().read_config()')
    cfg = read_config(args.config, args.filename)

    log.start('main().set_up_filenames(**kwargs)')
    filenames = get_filenames(cfg)

    log.start('main().calc_predictors(**kwargs)')
    if cfg.performance_diagnostics is None or cfg.obs_id is None:
        performance_diagnostics = None
    else:
        performance_diagnostics = calc_performance(filenames, cfg)
    independence_diagnostics = calc_independence(filenames, cfg)

    log.start('main().calc_deltas(**kwargs)')
    delta_q, delta_i, sigma_i_variants = calc_deltas(performance_diagnostics, independence_diagnostics, cfg)

    if cfg.target_diagnostic is None:
        logger.info('Using user sigmas: q={}, i={}'.format(cfg.sigma_q, cfg.sigma_i))
        sigma_q = cfg.sigma_q
        sigma_i = cfg.sigma_i
        targets = None
        clim = None
    else:
        log.start('main().calc_target(**kwargs)')
        targets, clim = calc_target(filenames[cfg.target_diagnostic], cfg)
        log.start('main().calc_sigmas(**kwargs)')
        sigma_q, sigma_i = calc_sigmas(targets, delta_i, sigma_i_variants, cfg)

    log.start('main().calc_weights(**kwargs)')
    weights = calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)

    log.start('main().save_data(**kwargs)')
    save_data(weights, targets, clim, filenames, cfg)
    log.stop

    if cfg.plot:
        logger.info('Plots are at: {}'.format(
            os.path.join(cfg.plot_path, cfg.config)))


if __name__ == "__main__":
    args = read_args()
    utils.set_logger(level=args.log_level, filename=args.log_file)
    with utils.LogTime(os.path.basename(__file__).replace('py', 'main()')):
        main(args)
