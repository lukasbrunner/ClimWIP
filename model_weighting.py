#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-01-28 09:47:02 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors
-------
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract
--------
Main script of the model weighting scheme described by Ruth et al.
2018 and Knutti et al. 2017. Reads a configuration file (default:
./configs/config.ini) and calculates target and predictor diagnostics. Target
diagnostics are used for a perfect model test, predictors for calculating the
weighting functions. Returns a combined quality-independence weight for each
included model.

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
"""
import os
import logging
import argparse
import numpy as np
import xarray as xr

import matplotlib as mpl
mpl.use('Agg')

from utils_python import utils
from utils_python.get_filenames import Filenames
from utils_python.xarray import add_hist, area_weighted_mean
from utils_python.decorators import vectorize

# I still don't understand how to properly do this :(
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
if __name__ == '__main__':
    from functions.diagnostics import calculate_diagnostic
    from functions.percentile import perfect_model_test
    from functions.weights import (
        calculate_weights_sigmas,
        calculate_weights,
        independence_sigma,
    )
    from functions.plots import (
        plot_rmse,
        plot_maps,
        plot_fraction_matrix,
        plot_weights,
    )
else:
    from model_weighting.functions.diagnostics import calculate_diagnostic
    from model_weighting.functions.percentile import perfect_model_test
    from model_weighting.functions.weights import (
        calculate_weights_sigmas,
        calculate_weights,
        independence_sigma,
    )
    from model_weighting.functions.plots import (
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
                   isinstance(cfg.debug, bool),
                   isinstance(cfg.plot, bool)]):
        raise ValueError('Typo in overwrite, debug, plot?')
    if cfg.ensemble_independence and not cfg.ensembles:
        raise ValueError('Can not use ensemble_independence without ensembles')
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


def read_config(args):
    cfg = utils.read_config(args.config, args.filename)
    utils.log_parser(cfg)
    test_config(cfg)
    return cfg


def get_filenames(fn, varn, all_members=True):
    """
    Return a list of filenames.

    Parameters
    ----------
    fn : class
        A Filenames class
    varn : str
        A valid variable name
    all_members=True : bool, optional
        If True include all available ensemble members per model. If False
        include only one (the first) member.

    Returns
    -------
    filenames : tuple
        A tuple of filenames
    """
    scenario = fn.get_variable_values('scenario')[0]

    model_ensemble, filenames = (), ()
    for model in fn.get_variable_values('model'):
        # remove ensemble members which are not available for all variables
        # ensemble members available for this model & variable
        ensembles_var = fn.get_variable_values(
            'ensemble', subset={'scenario': scenario,
                                'model': model,
                                'varn': varn})

        # ensemble members available for this model & all variables
        ensembles_all = fn.get_variable_values(
            'ensemble', subset={
                'scenario': scenario,
                'model': model,
                'varn': fn.get_variable_values('varn')})

        ensembles_select = []
        for ensemble in ensembles_var:
            if ensemble in ensembles_all:
                ensembles_select.append(ensemble)
            else:
                logmsg = ' '.join([
                    'Removed {} from {} (not available for all',
                    'variables)']).format(ensemble, model)
                logger.info(logmsg)

        assert len(ensembles_select) >= 1
        if not all_members:
            ensembles_select = ensembles_select[:1]
        for ensemble in ensembles_select:
            model_ensemble += ('{}_{}'.format(model, ensemble),)
            ff = fn.get_filenames(
                subset={'varn': varn,
                        'model': model,
                        'scenario': scenario,
                        'ensemble': ensemble})
            assert len(ff) == 1, 'len(ff) should be one!'
            filenames += (ff[0],)

    if len(filenames) < 20:
        # perfect model test will not work with too few models!
        logmsg = ' '.join([
            'Only {} files found (perfect model test will probably not work',
            'with too few models!']).format(len(filenames))
        logger.warning(logmsg)
    else:
        logger.info('{} files found.'.format(len(filenames)))
    logger.debug(', '.join(model_ensemble))
    return filenames, model_ensemble


def set_up_filenames(cfg):
    """Sets up the Filenames object.

    Add basic variables to create derived diagnostics to the list.

    Parameters
    ----------
    cfg : object
        A config object

    Returns
    -------
    fn : object
        A Filename object
    """
    varns = set([cfg.target_diagnostic] + cfg.predictor_diagnostics)

    # remove derived variables from original list and add base variables
    del_varns, add_varns = [], []
    for varn in varns:
        if varn in DERIVED.keys():
            del_varns.append(varn)
            add_varns += DERIVED[varn]
    varns = set(varns).difference(del_varns).union(add_varns)

    varns = list(varns)
    logger.info('Variables in analysis: {}'.format(', '.join(varns)))

    # set up filenames class
    fn = Filenames(
        file_pattern='{varn}/{varn}_{freq}_{model}_{scenario}_{ensemble}_g025.nc',
        base_path=cfg.data_path)
    fn.apply_filter(varn=varns, freq=cfg.freq, scenario=cfg.scenario)
    assert len(fn.get_variable_values('scenario')) == 1

    # restrict to models which are available for all variables
    models = fn.get_variable_values('model', subset={'varn': varns})

    # only use user-set models
    if cfg.select_models is not None:
        if len(cfg.select_models) != len(
                set(cfg.select_models).intersection(models)):
            errmsg = ' '.join([
                'select_models is not None but not all given models contain',
                'all required variables ([{}] not in [{}])'.format(
                    ', '.join(cfg.select_models), ', '.join(models))])
            raise ValueError(errmsg)
        models = cfg.select_models

    fn.apply_filter(model=models)

    logger.info('{} models included in analysis'.format(len(models)))
    logger.debug('Models included in analysis: {}'.format(', '.join(models)))
    return fn


def calc_target(fn, cfg):
    """
    Calculates the target variable for each model.

    Parameters
    ----------
    fn: object
        A Filename object
    cfg : object
        A config object

    Returns
    -------
    targets : xarray.DataArray, shape (L, M, N)
        DataArray of targets depending on models, lat, lon
    """
    base_path = os.path.join(
        cfg.save_path, cfg.target_diagnostic, cfg.freq,
        'masked' if cfg.target_masko else 'unmasked')
    os.makedirs(base_path, exist_ok=True)

    targets = []
    for filename, model_ensemble in zip(*get_filenames(
            fn, cfg.target_diagnostic, cfg.ensembles)):

        with utils.LogTime(model_ensemble, level='debug'):

            target = calculate_diagnostic(
                filename, cfg.target_diagnostic, base_path,
                time_period=(
                    cfg.target_startyear,
                    cfg.target_endyear),
                season=cfg.target_season,
                time_aggregation=cfg.target_agg,
                mask_ocean=cfg.target_masko,
                region=cfg.region,
                overwrite=cfg.overwrite,
            )

            if cfg.target_startyear_ref is not None:
                target_hist = calculate_diagnostic(
                    filename, cfg.target_diagnostic, base_path,
                    time_period=(
                        cfg.target_startyear_ref,
                        cfg.target_endyear_ref),
                    season=cfg.target_season,
                    time_aggregation=cfg.target_agg,
                    mask_ocean=cfg.target_masko,
                    region=cfg.region,
                    overwrite=cfg.overwrite,
                )
                # change historical to future
                target[cfg.target_diagnostic] -= target_hist[cfg.target_diagnostic]

        target['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        targets.append(target)
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
    return xr.concat(targets, dim='model_ensemble')[cfg.target_diagnostic]


def calc_predictors(fn, cfg):
    """
    Calculate the predictor diagnostics.

    Calculate the predictor diagnostics for each model and the distance between
    each diagnostic and the observations (quality -- delta_q) as well as the
    distance between the diagnostics of each model (independence -- delta_i).

    Parameters
    ----------
    fn : object
        A Filename object
    cfg : object
        A config object

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

        base_path = os.path.join(
            cfg.save_path, diagn, cfg.freq,
            'masked' if cfg.predictor_masko[idx] else 'unmasked')
        os.makedirs(base_path, exist_ok=True)

        # if its a derived diagnostic: get first basic variable to get one of
        # the filenames (the others will be created by replacement)
        varn = DERIVED[diagn][0] if diagn in DERIVED.keys() else diagn

        diagnostics = []
        for filename, model_ensemble in zip(*get_filenames(
                fn, varn, cfg.ensembles)):

            with utils.LogTime(model_ensemble, level='debug'):

                if diagn in list(DERIVED.keys()):
                    diagn = {diagn: DERIVED[diagn]}

                diagnostic = calculate_diagnostic(
                    filename, diagn, base_path,
                    time_period=(
                        cfg.predictor_startyears[idx],
                        cfg.predictor_endyears[idx]),
                    season=cfg.predictor_seasons[idx],
                    time_aggregation=cfg.predictor_aggs[idx],
                    mask_ocean=cfg.predictor_masko[idx],
                    region=cfg.region,
                    overwrite=cfg.overwrite,
                )

            diagnostic['model_ensemble'] = xr.DataArray(
                [model_ensemble], dims='model_ensemble')
            diagnostics.append(diagnostic)
            logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
        diagnostics = xr.concat(diagnostics, dim='model_ensemble')  # merge to one Dataset

        # TODO: move this to after if cfg.obsdata
        # -> get mask from obs and apply same mask to models first!
        logger.debug('Calculate model independence matrix...')
        diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn
        diagnostics['rmse_models'] = xr.DataArray(
            np.zeros((len(diagnostics[diagn_key]), len(diagnostics[diagn_key]))) * np.nan,
            dims=('model_ensemble', 'model_ensemble'))
        for ii, diagnostic1 in enumerate(diagnostics[diagn_key]):
            for jj, diagnostic2 in enumerate(diagnostics[diagn_key]):
                if ii == jj:
                    diagnostics['rmse_models'].data[ii, ii] = np.nan
                elif ii > jj:  # the matrix is symmetric
                    diagnostics['rmse_models'].data[ii, jj] = diagnostics['rmse_models'].data[jj, ii]
                else:
                    diff = area_weighted_mean((diagnostic1 - diagnostic2)**2)
                    if cfg.predictor_aggs[idx] == 'CYC':
                        diff = diff.sum('month')
                    diagnostics['rmse_models'].data[ii, jj] = np.sqrt(diff)
        logger.debug('Calculate independence matrix... DONE')

        if cfg.obsdata is not None:
            logger.debug('Read observations & calculate model quality...')

            filename = os.path.join(
                cfg.obs_path, '{}_mon_{}_g025.nc'.format(
                    varn, cfg.obsdata))

            with utils.LogTime('Calculate diagnostic for observations', level='debug'):
                obs = calculate_diagnostic(
                    filename, diagn, base_path,
                    time_period=(
                        cfg.predictor_startyears[idx],
                        cfg.predictor_endyears[idx]),
                    season=cfg.predictor_seasons[idx],
                    time_aggregation=cfg.predictor_aggs[idx],
                    mask_ocean=cfg.predictor_masko[idx],
                    region=cfg.region,
                    overwrite=cfg.overwrite,
                    regrid=cfg.obsdata in REGRID_OBS,
                )

            diff = diagnostics[diagn_key] - obs[diagn_key]

            try:
                cfg.obsdata_spread
            except AttributeError:
                cfg.obsdata_spread = None
            if cfg.obsdata_spread is not None:
                filename = os.path.join(
                    cfg.obs_path, '{}_mon_{}_g025_spread.nc'.format(
                        varn, cfg.obsdata))

                with utils.LogTime('Calculate diagnostic for observations', level='debug'):
                    obs_spread = calculate_diagnostic(
                        filename, diagn, base_path,
                        time_period=(
                            cfg.predictor_startyears[idx],
                            cfg.predictor_endyears[idx]),
                        season=cfg.predictor_seasons[idx],
                        time_aggregation=cfg.predictor_aggs[idx],
                        mask_ocean=cfg.predictor_masko[idx],
                        region=cfg.region,
                        overwrite=cfg.overwrite,
                        regrid=True,
                    )[diagn_key]

                @vectorize('(n,m),(n,m)->(n,m)')
                def correct_for_spread(data, spread):
                    """Correct for the spread in the observations.

                    Set differences inside of the spread to zero and move all
                    other differences so that they use the spread boundaries as
                    reference (instead of the mean).

                    Info
                    ----
                    old: -8  -4   0123456789 <- distances
                          o   |---x-o-|o   o <- o...model; x...obs; |...spread
                    new: -4   00000000012345 <- new distances
                    """
                    spread = .5*spread
                    not_significant = np.abs(data) <= spread
                    data = data - np.sign(data)*spread
                    data[not_significant] = 0
                    return data

                diff = xr.apply_ufunc(
                    correct_for_spread, diff, obs_spread,
                    input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                    output_core_dims=[['lat', 'lon']])

            diff = area_weighted_mean(diff**2)
            if cfg.predictor_aggs[idx] == 'CYC':
                diff = diff.sum('month')
            diagnostics['rmse_obs'] = np.sqrt(diff)
            logger.debug('Read observations & calculate model quality... DONE')

        logger.debug('Normalize data...')

        normalizer = diagnostics['rmse_models'].data
        if cfg.obsdata:
            # TODO: the difference in including this is probably minor
            # think about what it actually means to include this here
            normalizer = np.concatenate([normalizer, [diagnostics['rmse_obs'].data]], axis=0)

        if cfg.performance_normalize is None:
            normalizer = 1.
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
        else:
            raise ValueError

        if cfg.performance_normalize.lower() != 'map':
            diagnostics['rmse_models'] /= normalizer
            if cfg.obsdata:
                diagnostics['rmse_obs'] /= normalizer

        logger.debug('Normalize data... DONE')
        diagnostics_all.append(diagnostics)
        logger.info(f'Calculate diagnostic {diagn_key}{cfg.predictor_aggs[idx]}... DONE')
        # --- optional plot output for consistency checks ---
        if cfg.plot:
            with utils.LogTime('Plotting', level='info'):
                plotn = plot_rmse(diagnostics['rmse_models'], idx, cfg,
                                  diagnostics['rmse_obs'] if cfg.obsdata else None)
                if cfg.obsdata:
                    plot_maps(diagnostics, idx, cfg, obs=obs)
                else:
                    plot_maps(diagnostics, idx, cfg)

                add_hist(diagnostics)
                diagnostics.to_netcdf(plotn + '.nc')  # also save the data
                logger.debug('Saved plot data: {}.nc'.format(plotn))
        # ---------------------------------------------------

    # take the mean over all diagnostics and write them into a now Dataset
    # TODO: somehow xr.concat(diganostics_all) does not work -> fix it?
    delta_i = np.mean([dd['rmse_models'].data for dd in diagnostics_all], axis=0)
    if cfg.obsdata:
        delta_q = np.mean([dd['rmse_obs'].data for dd in diagnostics_all], axis=0)
        delta_q = xr.Dataset(
            coords={'model_ensemble': diagnostics['model_ensemble']},
            data_vars={'delta_q': ('model_ensemble', delta_q)})
    else:  # if there are not observations delta_i and delta_q are identical!
        delta_q = xr.Dataset(
            coords={
                'perfect_model_ensemble': diagnostics['model_ensemble'].data,
                'model_ensemble': diagnostics['model_ensemble'].data},
            data_vars={'delta_q': (('perfect_model_ensemble', 'model_ensemble'), delta_i)})

    delta_i = xr.Dataset(
        coords={
            'perfect_model_ensemble': diagnostics['model_ensemble'].data,
            'model_ensemble': diagnostics['model_ensemble'].data},
        data_vars={'delta_i': (('perfect_model_ensemble', 'model_ensemble'), delta_i)})

    # --- optional plot output for consistency checks ---
    if cfg.plot:
        plotn = plot_rmse(delta_i['delta_i'], 'mean', cfg,
                          delta_q['delta_q'] if cfg.obsdata else None)

        add_hist(diagnostics)
        diagnostics.to_netcdf(plotn + '.nc')  # also save the data
        logger.debug('Saved plot data: {}.nc'.format(plotn))
    # ---------------------------------------------------

    return delta_q['delta_q'], delta_i['delta_i']


def calc_sigmas(targets, delta_i, cfg, debug=False):
    """
    Perform a perfect model test to estimate the optimal shape parameters.

    Perform a perfect model test to estimate the optimal shape parameters for
    model quality and independence. See, e.g., Knutti et al., 2017.

    Parameters
    ----------
    targets : xarray.DataArray, shape (L, M, N)
        Array of targets. Should depend on models, lat, lon
    delta_i : xarray.DataArray, shape (L, L)
        Array of distances from each model to each other model
    fn : object
        A Filename object
    cfg : object
        A config object
    debug : bool, optional
        If True return weights_sigmas matrix as intermediate result.

    Returns
    -------
    sigma_q : float
        Optimal shape parameter for quality weighing
    sigma_i : float
        Optimal shape parameter for independence weighting
    """
    if cfg.sigma_i is not None or cfg.sigma_q is not None:
        logger.info('Using user sigmas: q={}, i={}'.format(cfg.sigma_q, cfg.sigma_i))
        return cfg.sigma_q, cfg.sigma_i

    SIGMA_SIZE = 41
    tmp = np.nanmean(delta_i)

    # a large value means all models have equal quality -> we want this as small as possible
    sigmas_q = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    # a large value means all models depend on each other, a small value means all models
    # are independent -> we want this ~delta_i
    sigmas_i = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)

    model_ensemble = targets['model_ensemble'].data
    models = [*map(lambda x: x.split('_')[0], model_ensemble)]
    _, idx, counts = np.unique(models, return_index=True, return_counts=True)
    model_ensemble_1ens = model_ensemble[idx]  # unique models

    targets_1ens = targets.sel(model_ensemble=model_ensemble_1ens)
    delta_i_1ens = delta_i.sel(model_ensemble=model_ensemble_1ens).sel(
        perfect_model_ensemble=model_ensemble_1ens).data
    targets_1ens_mean = area_weighted_mean(targets_1ens, latn='lat', lonn='lon').data

    idx_i_min = None
    if cfg.ensemble_independence:
        weighting_ratio = independence_sigma(delta_i, sigmas_i, idx, counts)
        idx_i_min = np.argmin(np.abs(weighting_ratio - 1))

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    if debug:  # DEBUG: intermediate result for testing
        return weights_sigmas

    # ratio of perfect models inside their respective weighted percentiles
    # for each sigma combination
    inside_ratio = perfect_model_test(
        targets_1ens_mean, weights_sigmas,
        perc_lower=cfg.percentiles[0],
        perc_upper=cfg.percentiles[1])

    if cfg.inside_ratio is None:
        cfg.inside_ratio = cfg.percentiles[1] - cfg.percentiles[0]
    inside_ok = inside_ratio >= cfg.inside_ratio

    if not np.any(inside_ok[:, idx_i_min]):
        logmsg = f'Perfect model test failed ({inside_ratio[:, idx_i_min].max():.4f} < {cfg.inside_ratio:.4f})!'
        raise ValueError(logmsg)
        # NOTE: force a result (probably not recommended?)
        # inside_ok = inside_ratio >= np.max(inside_ratio[:, idx_i_min])
        # logmsg += ' Setting inside_ratio to max: {}'.format(
        #     np.max(inside_ratio[:, idx_i_min]))
        # logger.warning(logmsg)

    if idx_i_min is not None:
        idx_q_min = np.argmin(1-inside_ok[:, idx_i_min])
    else:
        # in this matrix (i, j) find the element with the smallest sum i+j
        # which is True
        # NOTE: this is only correct if sigmas_q == sigmas_i
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
    cfg : object
        A config object.

    Returns
    -------
    weights : xarray.Dataset, same shape as delta_q
        An array of weights
    """
    if cfg.obsdata:
        numerator, denominator = calculate_weights(delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= weights.sum()
    else:  # in this case delta_q is a matrix for each model as truth once
        calculate_weights_matrix = np.vectorize(
            calculate_weights, signature='(n)->(n),(n)', excluded=[1, 2, 3])
        numerator, denominator = calculate_weights_matrix(delta_q, delta_i, sigma_q, sigma_i)
        weights = numerator/denominator
        weights /= np.nansum(weights, axis=-1)

    ds = delta_q.to_dataset().copy()
    ds = ds.rename({'delta_q': 'weights'})
    ds['weights'].data = weights
    ds['delta_q'] = delta_q
    ds['delta_i'] = delta_i
    ds['sigma_q'] = xr.DataArray([sigma_q])
    ds['sigma_i'] = xr.DataArray([sigma_i])

    if cfg.plot:
        plot_weights(ds, cfg, numerator, denominator)
        plot_weights(ds, cfg, numerator, denominator, sort=True)

    return ds


def save_data(ds, cfg, dtype='nc'):
    """Save the given Dataset to a file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save
    cfg : object
        A config object.
    dtype : {'nc', 'json'}, optional
        String giving a valid file type.

    Returns
    -------
    None
    """
    dtype = dtype.replace('.', '').lower()

    if dtype == 'nc':
        ds.attrs.update({'config': cfg.config, 'config_path': cfg.config_path})
        add_hist(ds)
        filename = os.path.join(cfg.save_path, '{}.nc'.format(cfg.config))
        ds.to_netcdf(filename)
        logger.info('Saved file: {}'.format(filename))
    elif dtype == 'json':
        raise NotImplementedError('Output for .json files not yet implemented')


def main(args):
    """Call functions"""
    log = utils.LogTime()

    log.start('main().read_config()')
    cfg = read_config(args)

    log.start('main().set_up_filenames(cfg)')
    fn = set_up_filenames(cfg)

    log.start('main().calc_target(fn, cfg)')
    targets = calc_target(fn, cfg)

    log.start('main().calc_predictors(fn, cfg)')
    delta_q, delta_i = calc_predictors(fn, cfg)

    log.start('main().calc_sigmas(targets, delta_i, cfg)')
    sigma_q, sigma_i = calc_sigmas(targets, delta_i, cfg)

    log.start('main().calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)')
    weights = calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)
    log.stop

    weights[cfg.target_diagnostic] = targets  # also save targets...
    # ... and the filenames of the targets
    temp = fn.get_filenames(
        subset={'varn': cfg.target_diagnostic},
        return_filters=['model', 'ensemble'])
    weights['filename'] = xr.DataArray(
        [ff for _, _, ff in temp],
        coords={'model_ensemble': [f'{mm}_{ee}' for mm, ee, _ in temp]},
        dims='model_ensemble')

    log.start('main().save_data(weights, cfg)')
    save_data(weights, cfg)
    log.stop

    if cfg.plot:
        logger.info('Plots are at: {}'.format(
            os.path.join(cfg.plot_path, cfg.config)))


if __name__ == "__main__":
    args = read_args()
    utils.set_logger(level=args.log_level, filename=args.log_file)
    with utils.LogTime(os.path.basename(__file__).replace('py', 'main()')):
        main(args)
