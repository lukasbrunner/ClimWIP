#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-10-16 18:12:58 lukbrunn>

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
import netCDF4 as nc

from utils_python import utils
# from utils_python.physics import area_weighted_mean
from utils_python.get_filenames import Filenames
from utils_python.xarray import add_hist, area_weighted_mean

from functions.diagnostics import calc_diag, calc_CORR
from functions.percentile import perfect_model_test
from functions.weights import calculate_weights_sigmas, calculate_weights

logger = logging.getLogger(__name__)


DERIVED = {
    'tasclt': ['clt', 'tas'],
    'rnet': ['rlus', 'rsds', 'rlds', 'rsus'],
    'ef': ['hfls', 'hfss'],
    'dtr': ['tasmax', 'tasmin']
}


def read_config():
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
    args = parser.parse_args()
    cfg = utils.read_config(args.config, args.filename)
    utils.log_parser(cfg)
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
        ensembles = fn.get_variable_values(
            'ensemble', subset={'scenario': scenario,
                                'model': model,
                                'varn': varn})

        # remove ensemble members which are not available for all variables
        ensembles_all = fn.get_variable_values(
            'ensemble', subset={
                'scenario': scenario,
                'model': model,
                'varn': fn.get_variable_values('varn')})
        for ensemble in ensembles:
            if ensemble not in ensembles_all:
                logmsg = ' '.join([
                    'Removed {} from {} (not available for all',
                    'variables)']).format(ensemble, model)
                logger.warning(logmsg)
                ensembles.remove(ensemble)

        assert len(ensembles) >= 1
        if not all_members:
            ensembles = ensembles[:1]
        for ensemble in ensembles:
            model_ensemble += ('{}_{}'.format(model, ensemble),)
            ff = fn.get_filenames(
                subset={'varn': varn,
                        'model': model,
                        'scenario': scenario,
                        'ensemble': ensemble})
            assert len(ff) == 1, 'len(ff) should be one!'
            filenames += (ff[0],)

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
    for varn in varns:
        if varn in DERIVED.keys():
            varns.remove(varn)
            for base_varn in DERIVED[varn]:
                varns.add(base_varn)

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
    targets : ndarray, shape (L, M, N)
        Array of targets depending on models, lat, lon
    """
    base_path = os.path.join(
        cfg.save_path, cfg.target_diagnostic, cfg.freq,
        'masked' if cfg.target_masko else 'unmasked')
    os.makedirs(base_path, exist_ok=True)

    targets = []
    for filename, model_ensemble in zip(*get_filenames(fn, cfg.target_diagnostic, cfg.ensembles)):
        logger.debug('Calculate diagnostics for {}...'.format(model_ensemble))
        filename_template = os.path.join(base_path, os.path.basename(filename))
        filename_template = filename_template.replace('.nc', '')

        filename_diag = calc_diag(infile=filename,
                                  outname=filename_template,
                                  diagnostic=cfg.target_diagnostic,
                                  masko=cfg.target_masko,
                                  syear=cfg.target_startyear,  # future
                                  eyear=cfg.target_endyear,
                                  season=cfg.target_season,
                                  kind=cfg.target_agg,
                                  region=cfg.region,
                                  overwrite=cfg.overwrite)
        target = xr.open_dataset(filename_diag)
        target = target.squeeze('time')

        if cfg.target_startyear_ref is not None:
            filename_diag = calc_diag(infile=filename,
                                      outname=filename_template,
                                      diagnostic=cfg.target_diagnostic,
                                      masko=cfg.target_masko,
                                      syear=cfg.target_startyear_ref,  # historical
                                      eyear=cfg.target_endyear_ref,
                                      season=cfg.target_season,
                                      kind=cfg.target_agg,
                                      region=cfg.region,
                                      overwrite=cfg.overwrite)

            target_hist = xr.open_dataset(filename_diag)
            target_hist = target_hist.squeeze('time')
            target[cfg.target_diagnostic] -= target_hist[cfg.target_diagnostic]

        target['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        targets.append(target)
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
    return xr.concat(targets, dim='model_ensemble')


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
    delta_q : ndarray, shape (N,)
        Array of distances from each model to the observations
        TODO: case with no observations!
    delta_i : ndarray, shape (N, N)
        Array of distances from each model to each other model
    lat : ndarray, shape (M,)
        Array of latitudes.
    lon : ndarray, shape (L,)
        Array of longitudes.
    """
    # for each file in filenames calculate all diagnostics for each time period
    diagnostics_all = []
    data_all = []
    for idx, diagn in enumerate(cfg.predictor_diagnostics):
        logger.info('Calculate diagnostics for {}...'.format(diagn))

        base_path = os.path.join(
            cfg.save_path, diagn, cfg.freq,
            'masked' if cfg.predictor_masko[idx] else 'unmasked')
        os.makedirs(base_path, exist_ok=True)

        derived = diagn in DERIVED.keys()
        if derived:
            varn = DERIVED[diagn][0]
        else:
            varn = diagn

        # if derived:
        #     filename_matrix = [get_filenames(fn, varn, cfg.ensembles)[0]
        #                        for varn in DERIVED[diagn]]

        diagnostics = []
        for filename, model_ensemble in zip(*get_filenames(fn, varn, cfg.ensembles)):
            logger.debug('Calculate diagnostics for {}...'.format(model_ensemble))

            if derived and diagn == 'tasclt':
                filename_diag = calc_CORR(infile=filename,
                                          base_path=base_path,
                                          variable1=varn,
                                          variable2='tas',
                                          masko=cfg.predictor_masko[idx],
                                          syear=cfg.predictor_startyears[idx],
                                          eyear=cfg.predictor_endyears[idx],
                                          season=cfg.predictor_seasons[idx],
                                          region=cfg.region,
                                          overwrite=cfg.overwrite)
            else:
                filename_template = os.path.join(
                    base_path, os.path.basename(filename))
                filename_template = filename_template.replace('.nc', '')

                filename_diag = calc_diag(infile=filename,
                                          outname=filename_template,
                                          diagnostic=diagn,
                                          variable=varn,
                                          masko=cfg.predictor_masko[idx],
                                          syear=cfg.predictor_startyears[idx],
                                          eyear=cfg.predictor_endyears[idx],
                                          season=cfg.predictor_seasons[idx],
                                          kind=cfg.predictor_aggs[idx],
                                          region=cfg.region,
                                          overwrite=cfg.overwrite)

            diagnostic = xr.open_dataset(filename_diag)
            diagnostic = diagnostic.squeeze('time')
            diagnostic['model_ensemble'] = xr.DataArray(
                [model_ensemble], dims='model_ensemble')
            diagnostics.append(diagnostic)
            logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
        diagnostics = xr.concat(diagnostics, dim='model_ensemble')  # merge to one Dataset

        # TODO: handle variable names in a more elegant way
        # TODO: this is connected to the handling of derived variables!
        varn = list(set(diagnostics.variables).difference(diagnostics.coords))
        assert len(varn) == 1
        varn = varn[0]

        logger.debug('Calculate model independence matrix...')
        diagnostics['rmse_models'] = xr.DataArray(
            np.empty((len(diagnostics[varn]), len(diagnostics[varn]))) * np.nan,
            dims=('model_ensemble', 'model_ensemble'))
        for ii, diagnostic1 in enumerate(diagnostics[varn]):
            for jj, diagnostic2 in enumerate(diagnostics[varn]):
                if ii == jj:
                    diagnostics['rmse_models'].data[ii, ii] = np.nan
                elif ii > jj:  # the matrix is symmetric
                    diagnostics['rmse_models'].data[ii, jj] = diagnostics['rmse_models'].data[jj, ii]
                else:
                    diagnostics['rmse_models'].data[ii, jj] = np.sqrt(area_weighted_mean(
                        (diagnostic1 - diagnostic2)**2))
        logger.debug('Calculate independence matrix... DONE')

        if cfg.obsdata is not None:
            logger.debug('Read observations & calculate model quality...')

            filename = os.path.join(
                cfg.obs_path, '{}_mon_{}_g025.nc'.format(
                    varn, cfg.obsdata))

            base_path = os.path.join(
                cfg.save_path, diagn, cfg.freq,
                'masked' if cfg.predictor_masko[idx] else 'unmasked')
            os.makedirs(base_path, exist_ok=True)
            filename_template = os.path.join(
                    base_path, os.path.basename(filename))
            filename_template = filename_template.replace('.nc', '')

            filename_obs = calc_diag(infile=filename,
                                     outname=filename_template,
                                     diagnostic=diagn,
                                     variable=varn,
                                     masko=cfg.predictor_masko[idx],
                                     syear=cfg.predictor_startyears[idx],
                                     eyear=cfg.predictor_endyears[idx],
                                     season=cfg.predictor_seasons[idx],
                                     kind=cfg.predictor_aggs[idx],
                                     region=cfg.region,
                                     overwrite=cfg.overwrite)

            obs = xr.open_dataset(filename_obs)
            obs = obs.squeeze('time')
            diagnostics['rmse_obs'] = np.sqrt(area_weighted_mean(
                    (diagnostics[varn] - obs[varn])**2))
            logger.debug('Read observations & calculate model quality... DONE')

        logger.debug('Normalize data...')

        norm = diagnostics['rmse_models'].data
        if cfg.obsdata:
            norm = np.concatenate([norm, [diagnostics['rmse_obs'].data]], axis=0)

        if cfg.performance_normalize.lower() == 'median':
            diagnostics['rmse_models'] /= np.nanmedian(norm)
            if cfg.obsdata:
                diagnostics['rmse_obs'] /= np.nanmedian(norm)
        elif cfg.performance_normalize.lower() == 'mean':
            diagnostics['rmse_models'] /= np.nanmean(norm)
            if cfg.obsdata:
                diagnostics['rmse_obs'] /= np.nanmean(norm)
        elif cfg.performance_normalize.lower() == 'map':  # TODO: needs testing!
            diagnostics['rmse_models'] = np.interp(
                 diagnostics['rmse_models'], [np.nanmin(norm), np.nanmax(norm)], [0, 1])
            if cfg.obsdata:
                diagnostics['rmse_obs'] = np.interp(
                    diagnostics['rmse_obs'], [np.nanmin(norm), np.nanmax(norm)], [0, 1])
        elif cfg.performance_normalize is not None:
            raise ValueError
        logger.debug('Normalize data... DONE')
        diagnostics_all.append(diagnostics)

    # import xarray as xr
    # models, ensembles = np.array(
    #     fn.get_filenames(subset={'varn': cfg.target_diagnostic},
    #                      return_filters=['model', 'ensemble'])).swapaxes(0, 1)[:2]
    # model_ensemble = ['{}_{}'.format(mm, ee) for mm, ee in zip(models, ensembles)]
    # ds = xr.Dataset(
    #     coords={'model_ensemble': model_ensemble,
    #             'model_ensemble2': model_ensemble},
    #     data_vars={
    #         'model': ('model_ensemble', models),
    #         'ensemble': ('model_ensemble', ensembles),
    #         'rmse_mod': (('diagnostic', 'model_ensemble', 'model_ensemble2'), d_delta_i),
    #         'rmse_obs': (('diagnostic', 'model_ensemble'), d_delta_q)})
    # ds.to_netcdf('./plot_scripts/rmse.nc')

    delta_i = np.mean([dd['rmse_models'].data for dd in diagnostics_all], axis=0)
    delta_i = xr.Dataset(
        coords={'model_ensemble': diagnostics['model_ensemble']},
        data_vars={'delta_i': (('model_ensemble', 'model_ensemble'), delta_i)})

    if cfg.obsdata:
        delta_q = np.mean([dd['rmse_obs'].data for dd in diagnostics_all], axis=0)
        delta_q = xr.Dataset(
            coords={'model_ensemble': diagnostics['model_ensemble']},
            data_vars={'delta_q': ('model_ensemble', delta_q)})
    else:  # if there are not observations delta_i and delta_q are identical!
        delta_q = xr.Dataset(
            coords={'model_ensemble': diagnostics['model_ensemble']},
            data_vars={'delta_q': (('model_ensemble', 'model_ensemble'), delta_q)})

    return delta_q['delta_q'].data, delta_i['delta_i'].data  # DEBUG


def calc_sigmas(targets, delta_i, lat, lon, fn, cfg, debug=False):
    """
    Perform a perfect model test to estimate the optimal shape parameters.

    Perform a perfect model test to estimate the optimal shape parameters for
    model quality and independence. See, e.g., Knutti et al., 2017.

    Parameters
    ----------
    targets : array_like, shape (L, M, N)
        Array of targets. Should depend on models, lat, lon
    delta_i : array_like, shape (L, L)
        Array of distances from each model to each other model
    lat : array_like, shape(M,)
        Array of latitudes.
    lon : array_like, shape (N,)
        Array of longitudes.
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
    SIGMA_SIZE = 41
    tmp = np.nanmean(delta_i)

    # a large value means all models have equal quality -> we want this as small as possible
    sigmas_q = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    # a large value means all models depend on each other, a small value means all models
    # are independent -> we want this ~delta_i
    # TODO, NOTE: maybe we want the sigma_i with the largest spread in weights?
    # in particular: the right sigma would deliver an about 10x higher value for denominator
    # in the case of a model with 10 members compared to a model with only one member
    # since we know that there is one model with 10 members, the larges element should be about
    # 10x the smallest one!
    sigmas_i = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    # sigmas_i = np.array([.45])  # DEBUG

    model_ensemble = targets['model_ensemble'].data
    models = [*map(lambda x: x.split('_')[0], model_ensemble)]
    _, idx = np.unique(models, return_index=True)  # index of unique models

    targets_1ens = targets.isel(model_ensemble=idx)
    delta_i_1ens = delta_i[idx, :][:, idx]
    targets_1ens_mean = area_weighted_mean(targets_1ens).data

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    if debug:  # DEBUG: intermediate result for testing
        return weights_sigmas

    # ratio of perfect models inside their respective weighted percentiles
    # for each sigma combination
    inside_ratio = perfect_model_test(targets_1ens_mean, weights_sigmas,
                                      perc_lower=cfg.percentiles[0],
                                      perc_upper=cfg.percentiles[1])

    inside_ok = inside_ratio >= cfg.inside_ratio

    # in this matrix (i, j) find the element with the smallest sum i+j
    # which is True
    # NOTE: this is only correct if sigmas_q == sigmas_i
    index_sum = 9999
    idx_i_min, idx_q_min = None, None
    for idx_q, qq in enumerate(inside_ok):
        if qq.sum() == 0:
            continue  # no fitting element
        elif idx_q >= index_sum:
            break  # no further optimization possible
        idx_i = np.where(qq)[0][0]
        if idx_i + idx_q < index_sum:
            index_sum = idx_i + idx_q
            idx_i_min, idx_q_min = idx_i, idx_q

    return sigmas_q[idx_q_min], sigmas_i[idx_i_min]


def calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg):
    """
    Calculate the weights for given set of parameters.

    Calculate the weights for given model qualities (delta_q), model
    independence (detal_i), and shape parameters sigma_q and sigma_i.

    Parameters
    ----------
    delta_q : array_like, shape (N,)
        Array of distances from each model to the observations.
    delta_i : array like, shape (N, N)
        Array of distances from each model to each other model.
    sigma_q : float
        Float giving the quality weighting shape parameter.
    sigma_i : float
        Float giving the independence weighting shape parameter.
    cfg : object
        A config object.

    Returns
    -------
    weights : ndarray, shape (N,)
        An array of weights
    """
    if cfg.obsdata:
        weights = calculate_weights(delta_q, delta_i, sigma_q, sigma_i)
        return weights / weights.sum()
    # Not sure in what case we would need that?
    raise NotImplementedError


def save_data(weights, fn, cfg, dtype='nc', data=None):
    """Save the given weights to a file.

    Parameters
    ----------
    weights : array_like
        Array of weights
    fn : object
        A Filename object.
    cfg : object
        A config object.
    dtype : {'nc', 'json'}, optional
        String giving a valid file type.
    data : array_like, optional
        TODO
    lat : array_like, optional
        TODO
    lon : array_like, optional
        TODO

    Information
    -----------
    dtype='nc'
        Size of weights needs to match the number of model-ensemble
        combinations in fn.

    Returns
    -------
    None
    """
    dtype = dtype.replace('.', '').lower()

    if dtype == 'nc':
        from xarray import Dataset, DataArray
        _, model_ensemble = get_filenames(fn, cfg.target_diagnostic, cfg.ensembles)
        ds = Dataset(
            coords={'model_ensemble': np.array(model_ensemble)},
            data_vars={'weights': ('model_ensemble', weights)},
            attrs={'config': cfg.config, 'config_path': cfg.config_path})
        if data is not None:
            ds[cfg.target_diagnostic] = data
        add_hist(ds)
        filename = os.path.join(cfg.save_path, '{}.nc'.format(cfg.config))
        ds.to_netcdf(filename)
        logger.info('Saved file: {}'.format(filename))
    elif dtype == 'json':
        raise NotImplementedError


def main():
    """Call functions"""
    utils.set_logger(level=logging.INFO)
    logger.info('Run program {}...'.format(os.path.basename(__file__)))

    logger.info('Load config file...')
    cfg = read_config()
    logger.info('Load config file... DONE')

    logger.info('Get filenames...')
    fn = set_up_filenames(cfg)
    logger.info('Get filenames... DONE')

    logger.info('Calculate target diagnostic...')
    targets = calc_target(fn, cfg)
    logger.info('Calculate target diagnostic... DONE')

    # DEBUG
    lat = targets['lat'].data
    lon = targets['lon'].data
    targets = targets[cfg.target_diagnostic]

    logger.info('Calculate predictor diagnostics and delta matrix...')
    delta_q, delta_i = calc_predictors(fn, cfg)
    logger.info('Calculate predictor diagnostics and delta matrix... DONE')

    if cfg.sigma_i is None or cfg.sigma_q is None:
        logger.info('Calculate sigmas...')
        sigma_q, sigma_i = calc_sigmas(targets, delta_i, lat, lon, fn, cfg)
        logger.info('Calculate sigmas... DONE')
    else:
        sigma_q, sigma_i = cfg.sigma_i, cfg.sigma_q
        logger.info('Using user sigmas: {}, {}'.format(sigma_i, sigma_q))

    logger.info('Calculate weights and weighted mean...')
    weights = calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)
    logger.info('Calculate weights and weighted mean... DONE')

    logger.info('Saving data...')
    save_data(weights, fn, cfg, data=targets)
    logger.info('Saving data... DONE')

    logger.info('Run program {}... Done'.format(os.path.basename(__file__)))


if __name__ == "__main__":
    main()
