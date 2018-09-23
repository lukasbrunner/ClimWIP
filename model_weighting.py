#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-23 18:50:22 lukas>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:
Main script of the model weighting scheme described by Ruth et al.
2018 and Knutti et al. 2017. Reads a configuration file (default:
./configs/config.ini) and calculates target and predictor diagnostics. Target
diagnostics are used for a perfect model test, predictors for calculating the
weighting functions. Returns a combined quality-independence weight for each
included model.

References:
Knutti, R., J. Sedláček, B. M. Sanderson, R. Lorenz, E. M. Fischer, and
V. Eyring (2017), A climate model projection weighting scheme accounting
for performance and interdependence, Geophys. Res. Lett., 44, 1909–1918,
 doi:10.1002/2016GL072012.

Lorenz, R., Herger, N., Sedláˇcek, J., Eyring, V., Fischer, E. M., and
Knutti, R. (2018). Prospects and caveats of weighting climate models for
summer maximum temperature projections over North America. Journal of
Geophysical Research: Atmospheres, 123, 4509–4526. doi:10.1029/2017JD027992.
"""
import os
import logging
import argparse
import numpy as np
import netCDF4 as nc

import utils_python.utils as utils
from utils_python.physics import area_weighted_mean
from utils_python.get_filenames import Filenames

from functions.diagnostics import calc_diag, calc_CORR
from functions.percentile import calculate_optimal_sigma
from functions.weights import calculate_weights_sigmas, calculate_weights

logger = logging.getLogger(__name__)


MAP_DIAGNOSTIC_VARN = dict(
    tasclt='clt',
    rnet='rlus',
    ef='hfls',
    dtr='tasmax'
)


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
    """Returns a list of filenames.

    Parameters:
    - fn (class): Filename class
    - varn (str): A valid variable name
    - all_members=True (bool, optional): If True include all available ensemble
      members per model. If False include only one (the first) member.

    Returns:
    tuple, tuple (identifiers, filenames)"""

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
                assert len(ff) == 1, 'len(ff) should be one!'
                filenames += (ff[0],)

    logger.info('{} files found.'.format(len(filenames)))
    logger.debug(', '.join(map(str, model_ens)))
    return filenames


def set_up_filenames(cfg):
    """Sets up the Filenames object. Adds basic variables to create derived
    diagnostics to the list.

    Parameters:
    cfg (config object)

    Returns:
    filename object
    """

    varns = set([cfg.target_diagnostic] + cfg.predictor_diagnostics)

    # remove derived variables from original list and add base variables
    if 'tasclt' in varns:  # DEBUG: this also needs 'tas' right???
        varns.remove('tasclt')
        varns.add('clt')
    if 'rnet' in varns:
        varns.remove('rnet')
        varns.add('rlus')
        varns.add('rsds')
        varns.add('rlds')
        varns.add('rsus')
    if 'ef' in varns:
        varns.remove('ef')
        varns.add('hfls')
        varns.add('hfss')
    if 'dtr' in varns:
        varns.remove('dtr')
        varns.add('tasmax')
        varns.add('tasmin')

    varns = list(varns)
    logger.info('Variables in analysis: {}'.format(', '.join(varns)))

    # set up filenames class
    fn = Filenames(
        file_pattern='{varn}/{varn}_{freq}_{model}_{scenario}_{ensemble}_g025.nc',
        base_path=cfg.data_path)
    fn.apply_filter(varn=varns, freq=cfg.freq, scenario=cfg.scenario)
    models = fn.get_variable_values('model')

    # DEBUG: exclude EC-EARTH for now
    if 'EC-EARTH' in models:
        models.remove('EC-EARTH')
        fn.apply_filter(model=models)

    # DEBUG: remove most models to speed up
    # models = models[:7]
    # fn.apply_filter(model=models)

    logger.info('{} models included in analysis'.format(len(models)))
    logger.debug('Models included in analysis: {}'.format(', '.join(models)))
    return fn


def calc_target(fn, cfg):
    """Calculates the target variable for each model.

    Parameters:
    - fn (Filename object):
    - cfg (config object):

    Returns:
    np.array of shape (len(models), len(lat), len(lon))"""

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
                                  syear=cfg.syear_fut,  # future
                                  eyear=cfg.eyear_fut,
                                  season=cfg.target_season,
                                  kind=cfg.target_agg,
                                  region=cfg.region,
                                  overwrite=cfg.overwrite)

        fh = nc.Dataset(filename_diag, mode='r')
        target = fh.variables[cfg.target_diagnostic][:]  # time, lat, lon
        fh.close()

        if cfg.target_type == 'change':
            filename_diag = calc_diag(infile=filename,
                                      outname=filename_template,
                                      diagnostic=cfg.target_diagnostic,
                                      masko=cfg.target_masko,
                                      syear=cfg.syear_hist,  # historical
                                      eyear=cfg.eyear_hist,
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
    return np.array(targets)


def calc_predictors(fn, cfg):
    """Calculate the predictor diagnostics for each model and the distance
    between each diagnostic and the observations (quality - delta_q) as well
    as the distance between the diagnostics of each model (independence -
    delta_i).

    Parameters:
    - fn (Filename object):
    - cfg (config object):

    Returns:
    delta_q, delta_i, lat, lon"""

    # for each file in filenames calculate all diagnostics for each time period
    rmse_all = []
    d_delta_i, d_delta_q = [], []
    lat, lon = None, None
    for idx, (diagn, agg, masko) in enumerate(
            zip(cfg.predictor_diagnostics,
                cfg.predictor_aggs,
                cfg.predictor_masko)):
        logger.info('Calculate diagnostics for {}...'.format(diagn))

        if cfg.predictor_derived[idx]:
            if diagn in MAP_DIAGNOSTIC_VARN.keys():
                varn = MAP_DIAGNOSTIC_VARN[diagn]
            else:
                logger.error('Unknown derived diagnostic.')
        else:
            varn = diagn

        base_path = os.path.join(
            cfg.save_path, diagn, cfg.freq,
            'masked' if cfg.target_masko else 'unmasked')
        os.makedirs(base_path, exist_ok=True)

        diagnostics = []
        for filename in get_filenames(fn, varn, cfg.ensembles):
            logger.debug('Calculate diagnostics for file {}...'.format(filename))

            # DEBUG: I don't understand the special handling of 'tasclt'
            # if ((row['derived'] == True and diag == 'tasclt')):
            #     outfile = '%s%s%s' %(diagdir[diag], 'tas', file_start)
            # else:
            #     outfile = '%s%s' %(diagdir[diag], file_start)
            if cfg.predictor_derived[idx] and diagn == 'tasclt':
                filename_diag = calc_CORR(infile=filename,
                                          base_path=base_path,
                                          variable1=varn,
                                          variable2='tas',
                                          masko=masko,
                                          syear=cfg.syear_eval[idx],
                                          eyear=cfg.eyear_eval[idx],
                                          season=cfg.predictor_seasons[idx],
                                          region=cfg.region)
            else:
                filename_template = os.path.join(
                    base_path, os.path.basename(filename))
                filename_template = filename_template.replace('.nc', '')

                filename_diag = calc_diag(infile=filename,
                                          outname=filename_template,
                                          diagnostic=diagn,
                                          variable=varn,
                                          masko=masko,
                                          syear=cfg.syear_eval[idx],
                                          eyear=cfg.eyear_eval[idx],
                                          season=cfg.predictor_seasons[idx],
                                          kind=agg,
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
                    rmse_models[ii, ii] = np.nan
                elif ii > jj:  # the matrix is symmetric
                    rmse_models[ii, jj] = rmse_models[jj, ii]
                else:
                    rmse_models[ii, jj] = np.sqrt(area_weighted_mean(
                        (diagnostic1 - diagnostic2)**2, lat, lon))
        logger.debug('Calculate independence matrix... DONE')

        if cfg.obsdata is not None:
            logger.debug('Read observations & calculate model quality...')

            # DEBUG: calculate on the fly or find file here!
            filename_obs = '%s%s_%s_%s_%s-%s_%s_%s_%s.nc' %(
                diagdir[diag], diag, cfg.freq, cfg.obsdata, cfg.syear_eval[i],
                cfg.eyear_eval[i], row['res_name'], row['var_file'], cfg.region)
            if not (os.path.isfile(filename_obs)):
                logger.error('Cannot find obs file %s, exiting.' %filename_obs)
                raise IOError

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
            rmse = np.concatenate((rmse_models, rmse_obs), axis=0)
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

    delta_i = np.array(d_delta_i).mean(axis=0)  # mean over all diagnostics
    if cfg.obsdata:
        delta_q = np.array(d_delta_q).mean(axis=0)  # mean over all diagnostics
    else:  # if there are not observations delta_i and delta_q are identical!
        delta_q = delta_i

    return delta_q, delta_i, lat, lon


def calc_sigmas(targets, delta_i, lat, lon, fn, cfg, debug=False):
    """Performs a perfect model test (e.g., Knutti et al., 2017;
    DOI: 10.1002/2016GL072012) to find the optimal weighting for model
    quality and independence.

    Parameters:
    - targets (np.array): 3D array (len(models), len(lat), len(lon))
    - delta_i (np.array): 2D array (len(models), len(models))
    - lat, lon (np.array): 1D arrays
    - fn (Filename object):
    - cfg (config object):
    - debug=False (bool, optional): If True return weights_sigmas matrix as
      intermediate result.

    Returns:
    sigma_q, sigma_i (floats, optimal sigmas)"""

    targets = targets.squeeze()

    SIGMA_SIZE = 41
    tmp = np.nanmean(delta_i)  # DEBUG

    # a large value means all models have equal quality -> we want this as small as possible
    sigmas_q = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    # sigmas_q = np.linspace(.05, 1, 41)
    # a large value means all models depend on each other, a small value means all models
    # are independent -> we want this ~delta_i
    # NOTE: use multiple ensemble members to determine this?!
    sigmas_i = np.linspace(.1*tmp, 1.9*tmp, SIGMA_SIZE)
    # sigmas_i = [.5]

    models = np.array(
        fn.get_filenames(subset={'varn': cfg.target_diagnostic},
                         return_filters='model')).swapaxes(0, 1)[0]
    _, idx = np.unique(models, return_index=True)  # index of unique models
    targets_1ens = targets[idx]
    delta_i_1ens = delta_i[idx, :][:, idx]
    targets_1ens_mean = area_weighted_mean(targets_1ens, lat, lon)

    weights_sigmas = calculate_weights_sigmas(delta_i_1ens, sigmas_q, sigmas_i)

    if debug:
        return weights_sigmas

    # DEBUG: remove inside_ratio
    # NOTE: this is only correct if sigmas_q == sigmas_i!!
    idx_q, idx_i, inside_ratio = calculate_optimal_sigma(
        targets_1ens_mean, weights_sigmas)
    import ipdb; ipdb.set_trace()

    # DEBUG: I could not yet reproduce this result!!
    return sigmas_q[idx_q], sigmas_i[idx_i]


def calc_weights(delta_i, delta_q, cfg):
    """TODO: docstring"""

    if cfg.obsdata:
        return calculate_weights(delta_q, delta_i, cfg.sigma_q, cfg.sigma_i)
    else:
        raise NotImplementedError


def save_data():
    """TODO: docstring"""
    raise NotImplementedError


def main():
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
    weights = calc_weights(delta_i, delta_q, cfg)
    # mean = np.ma.average(np.ma.masked_invalid(targets),
    #                      weights=weights, axis=0).filled_invalid(np.nan)
    logger.info('Calculate weights and weighted mean... DONE')

    logger.info('Saving data...')
    save_data()
    logger.info('Saving data...')

    logger.info('Run program {}... Done'.format(os.path.basename(__file__)))


if __name__ == "__main__":
    main()
