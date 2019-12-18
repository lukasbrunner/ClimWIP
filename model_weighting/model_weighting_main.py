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

from core.get_filenames import get_filenames
from core.diagnostics import calculate_diagnostic
from core.perfect_model_test import perfect_model_test
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


def test_config(cfg):
    """Some basic consistency tests for the config input"""
    if len({len(cfg[key]) for key in cfg.keys() if 'predictor' in key}) != 1:
        errmsg = 'All predictor_* variables need to have same length'
        raise ValueError(errmsg)
    if not os.access(cfg.save_path, os.W_OK | os.X_OK):
        raise ValueError('save_path is not writable')
    if (cfg.plot_path is not None and
        not os.access(cfg.plot_path, os.W_OK | os.X_OK)):
        raise ValueError('plot_path is not writable')
    if not np.all([isinstance(cfg.overwrite, bool),
                   isinstance(cfg.plot, bool)]):
        raise ValueError('Typo in overwrite, debug, plot?')
    if cfg.ensemble_independence and not cfg.ensembles:
        raise ValueError('Can not use ensemble_independence without ensembles')
    if cfg.sigma_i is not None:
        cfg.sigma_i = float(cfg.sigma_i)
    if cfg.sigma_q is not None:
        cfg.sigma_q = float(cfg.sigma_q)
    try:
        cfg.obs_uncertainty
    except AttributeError:
        cfg.obs_uncertainty = 'center'
    if (not isinstance(cfg['target_masko'], bool) or
        not np.all([isinstance(masko, bool) for masko in cfg['predictor_masko']])):
        errmsg = 'masko must be bool!'
        raise ValueError(errmsg)

    if cfg.obs_id is not None and cfg.obs_path is not None:
        if isinstance(cfg.obs_id, str):
            cfg.obs_id = [cfg.obs_id]
        if isinstance(cfg.obs_path, str):
            cfg.obs_path = [cfg.obs_path]
        if len(cfg.obs_id) != len(cfg.obs_path):
            errmsg = 'obs_id and obs_path need to have same length!'
            logger.error(errmsg)
            raise ValueError(errmsg)

    try:
        cfg.idx_lats = np.atleast_1d(cfg.idx_lats)
        cfg.idx_lons = np.atleast_1d(cfg.idx_lons)
    except AttributeError:
        cfg.idx_lats = None
        cfg.idx_lons = None

    if cfg.target_diagnostic is None and not (
            cfg.sigma_q is None and cfg.sigma_i is None):
        errmsg = 'If target_diagnostic is None, both sigmas need to be set!'
        logger.error(errmsg)
        raise ValueError(errmsg)

    if isinstance(cfg.model_path, str):
        cfg.model_path = [cfg.model_path]
    if isinstance(cfg.model_id, str):
        cfg.model_id = [cfg.model_id]
    if isinstance(cfg.model_scenario, str):
        cfg.model_scenario = [cfg.model_scenario]
    if len({len(cfg[key]) for key in cfg.keys() if 'model_' in key}) != 1:
        errmsg = 'All model_* variables need to have same length'
        raise ValueError(errmsg)

    try:
        cfg.subset
    except AttributeError:
        cfg.subset = None

    return None


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


def read_config(config, config_file):
    """Read a configuration from a configuration file.

    Parameters
    ----------
    config : string
        A string identifying a configuration in config_file. This string will
        also be used to name the final output file.
    config_file : string
        A valid configuration file name (ending with ".ini")

    Returns
    -------
    cfg : configuration object
        A configuration object with must contain all mandatory configurations.
    """
    cfg = utils.read_config(config, config_file)
    utils.log_parser(cfg)
    test_config(cfg)
    return cfg


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
    clim = []
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
                mask_ocean=cfg.target_masko,
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
                    mask_ocean=cfg.target_masko,
                    region=cfg.target_region,
                    overwrite=cfg.overwrite,
                    idx_lats=cfg.idx_lats,
                    idx_lons=cfg.idx_lons,
                )
                target[cfg.target_diagnostic] -= target_hist[cfg.target_diagnostic]
                target_hist['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')

                clim.append(target_hist)

        target['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        targets.append(target)
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
    return (xr.concat(targets, dim='model_ensemble')[cfg.target_diagnostic],
            xr.concat(clim, dim='model_ensemble')[cfg.target_diagnostic]
            if cfg.target_startyear_ref is not None else None)


def calc_predictors(filenames, cfg):
    """
    Calculates the predictor diagnostics for each model.

    Calculate the predictor diagnostics for each model and the distance between
    each diagnostic and the observations (quality -- delta_q) as well as the
    distance between the diagnostics of each model (independence -- delta_i).

    Parameters
    ----------
    filenames : nested dictionary
        See get_filenames() docstring for more information.
    cfg : configuration object
        See read_config() docstring for more information.

    Returns
    -------
    delta_q : xarray.DataArray, shape (N,) or (N, N)
        DataArray of distances from each model to the observations
    delta_i : xarray.DataArray, shape (N, N)
        DataArray of distances from each model to each other model

    Return Dimensions
    -----------------
    perfect_model_ensemble : shape (N,)
        The (model, ensemble) combination which is treated as 'perfect' for
        this case
    model_ensemble : (N,)
        The (model, ensemble) combination

    Both dimensions have equal values but they are treated formally different
    as xarray can not handle variables with the same dimension twice properly.
    """
    # for each file in filenames calculate all diagnostics for each time period
    diagnostics_all = []
    for idx, diagn in enumerate(cfg.predictor_diagnostics):
        logger.info(f'Calculate diagnostic {diagn}{cfg.predictor_aggs[idx]}...')

        base_path = os.path.join(cfg.save_path, diagn)
        os.makedirs(base_path, exist_ok=True)

        # if its a derived diagnostic: get first basic variable to get one of
        # the filenames (the others will be created by replacement)
        varn = DERIVED[diagn][0] if diagn in DERIVED.keys() else diagn

        diagnostics = []
        for model_ensemble, filename in filenames[varn].items():
            with utils.LogTime(model_ensemble, level='debug'):

                if diagn in list(DERIVED.keys()):
                    diagn = {diagn: DERIVED[diagn]}

                diagnostic = calculate_diagnostic(
                    infile=filename,
                    id_=model_ensemble.split('_')[2],
                    diagn=diagn,
                    base_path=base_path,
                    time_period=(
                        cfg.predictor_startyears[idx],
                        cfg.predictor_endyears[idx]),
                    season=cfg.predictor_seasons[idx],
                    time_aggregation=cfg.predictor_aggs[idx],
                    mask_ocean=cfg.predictor_masko[idx],
                    region=cfg.predictor_regions[idx],
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
        diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

        diagnostics['rmse_models'] = xr.apply_ufunc(
            weighted_distance_matrix, diagnostics[diagn_key],
            input_core_dims=[['model_ensemble', 'lat', 'lon']],
            output_core_dims=[['perfect_model_ensemble', 'model_ensemble']],
            kwargs={'lat': diagnostics['lat'].data},  # NOTE: comment out for unweighted
            vectorize=True,
        )

        if cfg.predictor_aggs[idx] == 'CYC':
            diagnostics['rmse_models'] = diagnostics['rmse_models'].mean('month')

        diagnostics['perfect_model_ensemble'] = diagnostics['model_ensemble'].data
        logger.debug('Calculate independence matrix... DONE')

        if cfg.obs_id is not None:
            logger.debug('Read observations & calculate model quality...')
            obs_list = []
            for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
                filename = os.path.join(obs_path, f'{varn}_mon_{obs_id}_g025.nc')

                with utils.LogTime(f'Calculate diagnostic for {obs_id}', level='info'):
                    obs = calculate_diagnostic(
                        infile=filename,
                        diagn=diagn,
                        base_path=base_path,
                        time_period=(
                            cfg.predictor_startyears[idx],
                            cfg.predictor_endyears[idx]),
                        season=cfg.predictor_seasons[idx],
                        time_aggregation=cfg.predictor_aggs[idx],
                        mask_ocean=cfg.predictor_masko[idx],
                        region=cfg.predictor_regions[idx],
                        overwrite=cfg.overwrite,
                        regrid=obs_id in REGRID_OBS,
                        idx_lats=cfg.idx_lats,
                        idx_lons=cfg.idx_lons,
                    )
                    obs_list.append(obs)

            obs = xr.concat(obs_list, dim='dataset_dim')

            # different methods of accounting for observational uncertainty
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                obs_min = obs.min('dataset_dim', skipna=False)
                obs_max = obs.max('dataset_dim', skipna=False)
            if cfg.obs_uncertainty == 'range':
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    diff = xr.apply_ufunc(
                        distance_uncertainty, diagnostics[diagn_key],
                        obs_min[diagn_key], obs_max[diagn_key],
                        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
                        output_core_dims=[['lat', 'lon']],
                        vectorize=True)
            elif cfg.obs_uncertainty == 'center':
                diff = diagnostics[diagn_key] - .5*(obs_min[diagn_key] + obs_max[diagn_key])
            elif cfg.obs_uncertainty == 'mean':
                diff = diagnostics[diagn_key] - obs.mean('dataset_dim', skipna=False)[diagn_key]
            elif cfg.obs_uncertainty == 'median':
                diff = diagnostics[diagn_key] - obs.median('dataset_dim', skipna=False)[diagn_key]
            elif cfg.obs_uncertainty is None:
                # obs_min and obs_max are the same for this case
                diff = diagnostics[diagn_key] - obs_min[diagn_key]
            else:
                raise NotImplementedError

            # --- plot map of each difference ---
            # takes a long time -> only commend in if needed
            # if cfg.plot:
            #     with utils.LogTime('Plotting maps', level='info'):
            #         plot_maps(diff, idx, cfg)
            # ---------------------------------------

            diff = np.sqrt(area_weighted_mean(diff**2))
            if cfg.predictor_aggs[idx] == 'CYC':
                diff = diff.mean('month')
            diagnostics['rmse_obs'] = diff
            logger.debug('Read observations & calculate model quality... DONE')

        logger.debug('Normalize data...')

        # different methods of normalizing each diagnostic
        normalizer = diagnostics['rmse_models'].data
        if cfg.obs_id is not None:
            normalizer = np.concatenate([normalizer, [diagnostics['rmse_obs'].data]], axis=0)

        if cfg.performance_normalize is None:
            normalizer = 1.
        elif cfg.performance_normalize.lower() == 'middle':
            normalizer = .5*(np.nanmin(normalizer) + np.nanmax(normalizer))
        elif cfg.performance_normalize.lower() == 'median':
            normalizer = np.nanmedian(normalizer)
        elif cfg.performance_normalize.lower() == 'mean':
            normalizer = np.nanmean(normalizer)
        elif cfg.performance_normalize.lower() == 'map':  # TODO: needs testing!
            diagnostics['rmse_models'].data = np.interp(
                diagnostics['rmse_models'], [np.nanmin(normalizer),
                                             np.nanmax(normalizer)], [0, 1])
            diagnostics['rmse_obs'].data = np.interp(
                diagnostics['rmse_obs'], [np.nanmin(normalizer),
                                          np.nanmax(normalizer)], [0, 1])
            normalizer = 1
        else:
            raise NotImplementedError

        diagnostics['rmse_models'] /= normalizer
        if cfg.obs_id is not None:
            diagnostics['rmse_obs'] /= normalizer

        logger.debug('Normalize data... DONE')
        diagnostics_all.append(diagnostics)
        logger.info(f'Calculate diagnostic {diagn_key}{cfg.predictor_aggs[idx]}... DONE')

        # --- optional: plot RMSE matrix ---
        if cfg.plot:
            plotn = plot_rmse(diagnostics['rmse_models'], idx, cfg,
                              diagnostics['rmse_obs'] if cfg.obs_id is not None else None)
            add_revision(diagnostics)
            # diagnostics.to_netcdf(plotn + '.nc')  # NOTE: also save the data
            logger.debug('Saved plot data: {}.nc'.format(plotn))
        # ----------------------------------

    # take the mean over all diagnostics and write them into a now Dataset
    # TODO: somehow xr.concat(diganostics_all) does not work -> fix it?
    delta_i = xr.concat(
        [dd['rmse_models'] for dd in diagnostics_all], dim='diagnostics')
    delta_i = delta_i.mean('diagnostics')
    delta_i.name = 'delta_i'

    if cfg.obs_id is not None:
        delta_q = xr.concat(
            [dd['rmse_obs'] for dd in diagnostics_all], dim='diagnostics')
        delta_q = delta_q.mean('diagnostics')
    else:  # if there are not observations delta_i and delta_q are identical!
        delta_q = delta_i.copy()
    delta_q.name = 'delta_q'

    # --- optional: plot mean RMSE matrix ---
    if cfg.plot:
        plotn = plot_rmse(delta_i, 'mean', cfg,
                          delta_q if cfg.obs_id is not None else None)
        add_revision(diagnostics)
        # diagnostics.to_netcdf(plotn + '.nc')  # also save the data
        logger.debug('Saved plot data: {}.nc'.format(plotn))
    # ---------------------------------------

    return delta_q, delta_i


def calc_sigmas(targets, delta_i, unique_models, cfg, n_sigmas=50):
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
    if cfg.sigma_q == -99.:  # convention: equal weighting
        sigmas_q = np.array([-99.])
    else:
        sigmas_q = np.linspace(.2*sigma_base, 2*sigma_base, n_sigmas)
    # a large value means all models depend on each other, a small value means all models
    # are independent -> we want this ~delta_i
    if cfg.sigma_i == -99.:  # convention: equal weighting
        sigmas_i = np.array([-99.])
    else:
        sigmas_i = np.linspace(.2*sigma_base, 2*sigma_base, n_sigmas)

    # for the perfect model test we only use one member per model!
    targets_1ens = targets.sel(model_ensemble=unique_models)
    delta_i_1ens = delta_i.sel(model_ensemble=unique_models,
                               perfect_model_ensemble=unique_models).data
    targets_1ens_mean = area_weighted_mean(targets_1ens, latn='lat', lonn='lon').data

    # use the initial-condition members to estimate sigma_i
    if cfg.ensemble_independence:
        sigmas_i = independence_sigma(delta_i, sigmas_i)
        idx_i_min = 0
    else:
        idx_i_min = None

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    # ratio of perfect models inside their respective weighted percentiles
    # for each sigma combination

    inside_ratio = perfect_model_test(
        targets_1ens_mean, weights_sigmas,
        perc_lower=cfg.percentiles[0],
        perc_upper=cfg.percentiles[1])

    force_inside_ratio = False
    if cfg.inside_ratio is not None and cfg.inside_ratio.lower() == 'force':
        force_inside_ratio = True
    if cfg.inside_ratio is None or force_inside_ratio:
        cfg.inside_ratio = cfg.percentiles[1] - cfg.percentiles[0]
    inside_ok = inside_ratio >= cfg.inside_ratio

    if not np.any(inside_ok):
        logmsg = f'Perfect model test failed ({inside_ratio.max():.4f} < {cfg.inside_ratio:.4f})!'
        if force_inside_ratio:
            # adjust inside_ratio to force a result (probably not recommended?)
            inside_ok = inside_ratio >= np.max(inside_ratio[:, idx_i_min])
            logmsg += ' force=True: Setting inside_ratio to max: {}'.format(
                np.max(inside_ratio[:, idx_i_min]))
            logger.warning(logmsg)
        else:
            raise ValueError(logmsg)

    if cfg.ensemble_independence:
        idx_q_min = np.argmin(1-inside_ok[:, idx_i_min])
    else:
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
    weights : xarray.Dataset, same shape as delta_q
        An array of weights
    """
    if cfg.obs_id is not None:
        numerator, denominator = calculate_weights(delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= weights.sum()
        dims = 'model_ensemble'
    else:  # in this case delta_q is a matrix for each model as truth once
        calculate_weights_matrix = np.vectorize(
            calculate_weights, signature='(n)->(n),(n)', excluded=[1, 2, 3])
        numerator, denominator = calculate_weights_matrix(delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= np.nansum(weights, axis=-1)
        dims = ('perfect_model_ensemble', 'model_ensemble')

    ds = delta_q.to_dataset().copy()
    ds = ds.rename({'delta_q': 'weights'})
    ds['weights'].data = weights
    ds['weights_q'] = xr.DataArray(numerator, dims=dims)
    ds['weights_i'] = xr.DataArray(denominator, dims=dims)
    ds['delta_q'] = delta_q
    ds['delta_i'] = delta_i
    ds['sigma_q'] = xr.DataArray([sigma_q])
    ds['sigma_i'] = xr.DataArray([sigma_i])

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

    ds.attrs['target'] = cfg.target_diagnostic
    ds.attrs['region'] = cfg.target_region

    if cfg.plot and cfg.obs_id:
        plot_weights(ds, cfg, numerator, denominator)
        plot_weights(ds, cfg, numerator, denominator, sort=True)

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
    # save additional variables for convenience
    if targets is not None:
        ds[cfg.target_diagnostic] = targets
    if clim is not None:
        ds[f'{cfg.target_diagnostic}_clim'] = clim
    ds['filename'] = xr.DataArray(
        [*filenames[cfg.target_diagnostic].values()],
        coords={'model_ensemble': [*filenames[cfg.target_diagnostic].keys()]},
        dims='model_ensemble',
        attrs={
            'units': '1',
            'long_name': 'Full Path and Filename',
        })

    # add some metadata
    ds.attrs.update({
        'config': cfg.config,
        'config_path': cfg.config_path,
        'reference': ' '.join([
            'Brunner et al. (2019): Quantifying uncertainty in European',
            'climate projections using combined performance-independence',
            'weighting. Eniron. Res. Lett.',
            'https://doi.org/10.1088/1748-9326/ab492f.']),
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

    log.start('main().set_up_filenames(cfg)')

    # get basic variables for diagnostics
    varns = []
    for varn in cfg.predictor_diagnostics:
        if varn in DERIVED:
            varns.append(DERIVED[varn][0])
            varns.append(DERIVED[varn][1])
        else:
            varns.append(varn)

    if cfg.target_diagnostic is not None:
        # only if sigmas are None we need to calculate the target
        varns += [cfg.target_diagnostic]

    varns = np.unique(varns)
    filenames, unique_models = get_filenames(
        varns, cfg.model_id, cfg.model_scenario, cfg.model_path, cfg.ensembles,
        subset=cfg.subset)

    log.start('main().calc_predictors(fn, cfg)')
    delta_q, delta_i = calc_predictors(filenames, cfg)

    if cfg.target_diagnostic is None:
        logger.info('Using user sigmas: q={}, i={}'.format(cfg.sigma_q, cfg.sigma_i))
        sigma_q = cfg.sigma_q
        sigma_i = cfg.sigma_i
        targets = None
        clim = None
    else:
        log.start('main().calc_target(fn, cfg)')
        targets, clim = calc_target(filenames[cfg.target_diagnostic], cfg)
        log.start('main().calc_sigmas(targets, delta_i, cfg)')
        sigma_q, sigma_i = calc_sigmas(targets, delta_i, unique_models, cfg)

    log.start('main().calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)')
    weights = calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)

    log.start('main().save_data(weights, cfg)')
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
