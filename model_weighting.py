#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-10-11 17:41:13 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors
-------
Ruth Lorenz || ruth.lorenz@env.ethz.ch
Lukas Brunner || lukas.brunner@env.ethz.ch

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
import netCDF4 as nc

from utils_python import utils
from utils_python.physics import area_weighted_mean
from utils_python.get_filenames import Filenames
from utils_python.xarray import add_hist

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
    model_ens, filenames = (), ()
    for model in fn.get_variable_values('model'):
        for scenario in fn.get_variable_values('scenario'):
            ensembles = fn.get_variable_values(
                'ensemble', subset={'scenario': scenario, 'model': model})
            if not all_members:
                ensembles = ensembles[:1]
            for ensemble in ensembles:
                model_ens += ('{}-{}'.format(model, ensemble),)
                ff = fn.get_filenames(
                    subset={'varn': varn,
                            'model': model,
                            'scenario': scenario,
                            'ensemble': ensemble})
                # if len(ff) != 1:  # DEBUG
                #     import ipdb; ipdb.set_trace()
                assert len(ff) == 1, 'len(ff) should be one!'
                filenames += (ff[0],)

    logger.info('{} files found.'.format(len(filenames)))
    logger.debug(', '.join(map(str, model_ens)))
    return filenames


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

    # restrict to models which are available for all variables
    models = fn.get_variable_values('model', subset={'varn': varns})

    # DEBUG: exclude EC-EARTH for now
    # (not all variables have the same ensemble members)
    if 'EC-EARTH' in models:
        models.remove('EC-EARTH')

    if cfg.debug:
        models = models[:10]

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
    for filename in get_filenames(fn, cfg.target_diagnostic, cfg.ensembles):
        logger.debug('Calculate diagnostics for file {}...'.format(filename))
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

        fh = nc.Dataset(filename_diag, mode='r')
        target = fh.variables[cfg.target_diagnostic][:]  # time, lat, lon
        fh.close()

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

            fh = nc.Dataset(filename_diag, mode='r')
            target_hist = fh.variables[cfg.target_diagnostic][:]  # time, lat, lon
            fh.close()
            target -= target_hist

        targets.append(np.ma.filled(target, fill_value=np.nan))
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))
    return np.array(targets).squeeze()


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
    rmse_all = []
    d_delta_i, d_delta_q = [], []
    lat, lon = None, None
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

        diagnostics = []
        for filename in get_filenames(fn, varn, cfg.ensembles):
            logger.debug('Calculate diagnostics for file {}...'.format(filename))

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

            # For each diagnostic, read data to calculate perfmetric
            fh = nc.Dataset(filename_diag, mode='r')
            try:
                diagnostic = fh.variables[diagn][:]  # time, lat, lon
            except KeyError:
                diagnostic = fh.variables[varn][:]  # time, lat, lon
            diagnostics.append(np.ma.filled(diagnostic, fill_value=np.nan))
            if lat is None:
                lat, lon = fh.variables['lat'][:], fh.variables['lon'][:]
            fh.close()

            logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

        logger.debug('Calculate model independence matrix...')
        rmse_models = np.empty((len(diagnostics), len(diagnostics))) * np.nan
        for ii, diagnostic1 in enumerate(diagnostics):
            for jj, diagnostic2 in enumerate(diagnostics):
                if ii == jj:
                    rmse_models[ii, ii] = 0.  # DEBUG: change to np.nan (rework tests will fail)
                elif ii > jj:  # the matrix is symmetric
                    rmse_models[ii, jj] = rmse_models[jj, ii]
                else:
                    rmse_models[ii, jj] = np.sqrt(area_weighted_mean(
                        (diagnostic1 - diagnostic2)**2, lat, lon))
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

            fh = nc.Dataset(filename_obs, mode='r')
            try:
                obs = fh.variables[diagn][:]
            except KeyError:
                obs = fh.variables[varn][:]
            fh.close()

            rmse_obs = np.empty(len(diagnostics)) * np.nan
            for ii, diagnostic in enumerate(diagnostics):
                rmse_obs[ii] = np.sqrt(area_weighted_mean(
                    (diagnostic - obs)**2, lat, lon))
            rmse = np.concatenate((rmse_models, [rmse_obs]), axis=0)
            logger.debug('Read observations & calculate model quality... DONE')
        else:
            rmse = rmse_models
        rmse_all.append(rmse)

        # normalize deltas by median
        med = np.nanmedian(rmse)
        d_delta_i.append(np.divide(rmse_models, med))
        if cfg.obsdata:
            # NOTE: is this really the right way to normalize this??
            d_delta_q.append(np.divide(rmse_obs, med))
        # TODO: maybe we want to do this:
        # map the values of d_delta from [d_delta.min(), d_delta.max()] to [0, 1]
        # rmse_models = np.interp(rmse_models, [np.nanmin(rmse_models), np.nanmax(rmse_models)], [0, 1])
        # rmse_obs = np.interp(rmse_obs, [np.nanmin(rmse_obs), np.nanmax(rmse_obs)], [0, 1])
        # d_delta_i.append(rmse_models)

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

    delta_i = np.array(d_delta_i).mean(axis=0)  # mean over all diagnostics
    if cfg.obsdata:
        delta_q = np.array(d_delta_q).mean(axis=0)  # mean over all diagnostics
    else:  # if there are not observations delta_i and delta_q are identical!
        delta_q = delta_i

    return delta_q, delta_i, lat, lon


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
    # sigmas_i = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    sigmas_i = np.array([.45])  # DEBUG

    models = np.array(
        fn.get_filenames(subset={'varn': cfg.target_diagnostic},
                         return_filters='model')).swapaxes(0, 1)[0]
    _, idx = np.unique(models, return_index=True)  # index of unique models
    targets_1ens = targets[idx]
    delta_i_1ens = delta_i[idx, :][:, idx]
    targets_1ens_mean = area_weighted_mean(targets_1ens, lat, lon)

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    if debug:  # DEBUG: intermediate result for testing
        return weights_sigmas

    # ratio of perfect models inside their respective weighted percentiles
    # for each sigma combination
    inside_ratio = perfect_model_test(targets_1ens_mean, weights_sigmas)

    inside_ok = inside_ratio >= .8  # NOTE: 80% of time

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


def save_data(weights, fn, cfg, dtype='nc', data=None, lat=None, lon=None):
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
        models, ensembles, _ = np.array(fn.get_filenames(
            subset={'varn': cfg.target_diagnostic},
            return_filters=['model', 'ensemble'])).swapaxes(0, 1)
        model_ensemble = [
            '{}_{}'.format(mm, ee) for mm, ee in zip(models, ensembles)]
        ds = Dataset(
            coords={
                'model_ensemble': model_ensemble},
            data_vars={
                'model': ('model_ensemble', models),
                'ensemble': ('model_ensemble', ensembles),
                'weights': ('model_ensemble', weights)},
            attrs={
                'config': cfg.config,
                'config_path': cfg.config_path})
        if data is not None:
            da = DataArray(
                data=data,
                coords={'model_ensemble': model_ensemble,
                        'lat': lat, 'lon': lon},
                dims=('model_ensemble', 'lat', 'lon'),
                name=cfg.target_diagnostic)
            ds[cfg.target_diagnostic] = da
            import ipdb; ipdb.set_trace()
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

    logger.info('Calculate predictor diagnostics and delta matrix...')
    delta_q, delta_i, lat, lon = calc_predictors(fn, cfg)
    logger.info('Calculate predictor diagnostics and delta matrix... DONE')

    if cfg.sigma_type == "inpercentile":
        logger.info('Calculate sigmas...')
        sigma_q, sigma_i = calc_sigmas(targets, delta_i, lat, lon, fn, cfg)
        logger.info('Calculate sigmas... DONE')
    elif cfg.sigma_type == 'manual':
        sigma_q, sigma_i = cfg.sigma_i, cfg.sigma_q
        logger.info('Using user sigmas: {}, {}'.format(sigma_i, sigma_q))
    else:
        errmsg = ' '.join(['simga_type has to be one of [interpercentile |',
                           'manual] not {}'.format(cfg.sigma_type)])
        logger.error(errmsg)
        raise NotImplementedError(errmsg)

    logger.info('Calculate weights and weighted mean...')
    weights = calc_weights(delta_q, delta_i, sigma_q, sigma_i, cfg)
    logger.info('Calculate weights and weighted mean... DONE')

    logger.info('Saving data...')
    save_data(weights, fn, cfg, data=targets, lat=lat, lon=lon)
    logger.info('Saving data... DONE')

    logger.info('Run program {}... Done'.format(os.path.basename(__file__)))


if __name__ == "__main__":
    main()
