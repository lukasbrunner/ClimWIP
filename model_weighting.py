#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-19 15:39:32 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import json
import logging
import argparse
import numpy as np
import netCDF4 as nc
import multiprocessing as mp

import utils_python.utils as utils
from utils_python.run_parallel import run_parallel
from utils_python.get_filenames import Filenames

from functions.diagnostics import calc_diag, calc_CORR, calc_Rnet
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
        help='Name of the configuration to use.')
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
    - all_members=True (bool, optional): True: include all available ensemble members
      per model. False: include only one (the first) member

    Returns:
    tuple, tuple (identifiers, filenames)"""

    model_ens, filenames = (), ()
    for model in fn.get_variable_values('model'):
        for scenario in fn.get_variable_values('scenario'):
            ensembles = fn.get_variable_values(
                'ensemble',
                subset={'scenario': scenario, 'model': model})
            if not all_members:
                ensembles = ensembles[:1]
            for ensemble in ensembles:
                model_ens += ('{}-{}'.format(model, ensemble),)
                ff = fn.get_filenames(
                    subset={'varn': varn,
                            'model': model,
                            'scenario': scenario,
                            'ensemble': ensemble})
                if len(ff) != 1:
                    raise ValueError('This should be one!!')
                filenames += (ff[0],)

    logger.info('Number of included runs is {}'.format(len(filenames)))
    logger.debug('Included runs: {}'.format(', '.join(map(str, model_ens))))
    return model_ens, filenames


def set_up_filenames(cfg):
    """Sets up the Filenames object. Adds basic variables to create derived
    diagnostics to the list."""

    varns = set([cfg.target_diagnostic] + cfg.predictor_diagnostics)

    # remove derived variables from original list and add base variables
    if 'tasclt' in varns:  # TODO: this also needs 'tas' right???
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

    fn.apply_filter(varn=varns, freq=cfg.freq, scenario=cfg.scenario)  # restrict by variable and scenario
    models = fn.get_variable_values(
        'model', subset={'varn': varns, 'scenario': cfg.scenario})
    # DEBUG, TODO: exclude EC-EARTH for now
    if 'EC-EARTH' in models:
        models.remove('EC-EARTH')
        fn.apply_filter(model=models)

    # DEBUG: remove most models to speed up
    models = models[:7]
    fn.apply_filter(model=models)

    logger.info('Number of models included in analysis is: %s' %(len(models)))
    logger.debug('Models included in analysis are: %s' %(models))

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
    for model_ens, filename in zip(*get_filenames(
            fn, cfg.target_diagnostic, all_members=cfg.ensembles)):
        logger.debug('Calculate diagnostics for file {}...'.format(filename))
        # create template for output files
        filename_template = os.path.join(base_path, os.path.basename(filename))
        filename_template = filename_template.replace('.nc', '')

        filename_diag = calc_diag(filename, filename_template, cfg.target_diagnostic,
                                  masko=cfg.target_masko,
                                  syear=cfg.syear_fut,  # future
                                  eyear=cfg.eyear_fut,
                                  season=cfg.target_season,
                                  kind=cfg.target_agg,
                                  region=cfg.region,
                                  overwrite=cfg.overwrite)

        fh = nc.Dataset(filename_diag, mode = 'r')
        target = fh.variables[cfg.target_diagnostic][:] # time, lat, lon data
        fh.close()

        if cfg.target_type == 'change':
            filename_diag = calc_diag(filename, filename_template, cfg.target_diagnostic,
                                      masko=cfg.target_masko,
                                      syear=cfg.syear_hist,  # historical
                                      eyear=cfg.eyear_hist,
                                      season=cfg.target_season,
                                      kind=cfg.target_agg,
                                      region=cfg.region,
                                      overwrite=cfg.overwrite)

            fh = nc.Dataset(filename_diag, mode = 'r')
            target_hist = fh.variables[cfg.target_diagnostic][:] # time, lat, lon data
            fh.close()
            target -= target_hist

        targets.append(np.ma.filled(target, fill_value=np.nan))
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

    return np.array(targets)


def calc_predictors(fn, cfg):
    """Calculate the predictor diagnostics for each model.
    TODO: docstring"""

    # TODO: do this with the utils.weighted_mean function?!
    # like this: np.sqrt(utils.weighted_mean((data1-data2)**2, lat))
    def rmse_weighted(data1, data2, lat):
        # basic tests
        if len(data1.shape) == 2:
            data1 = data1.reshape(1, *data1.shape)
            data2 = data2.reshape(1, *data2.shape)
        elif len(data1.shape) != 3:
            errmsg = 'Shape of data1 needs to be (N, M) or (L, M, N) and not {}'.format(
                data1.shape)
            logger.error(errmsg)
            raise ValueError
        if data1.shape[1] != len(lat):
            errmsg = 'Shape of data1 does not fit given lat: {}!=(*, {}, *)'.format(
                data1.shape, len(lat))
            logger.error(errmsg)
            raise ValueError
        if data1.shape != data2.shape:
            errmsg = 'Shapes of data1 and data2 do not fit: {}!={}'.format(data1.shape, data2.shape)
            logger.error(errmsg)
            raise ValueError

        w_lat = np.cos(np.radians(lat))
        weights = np.tile(w_lat, (data1.shape[0], data1.shape[2], 1)).swapaxes(1, 2)
        data1, data2 = np.ma.masked_invalid(data1), np.ma.masked_invalid(data2)
        return np.sqrt(np.ma.average((data1-data2)**2, weights=weights))
    # --- /rmse_weighted/ ---

    # for each file in filenames calculate all diagnostics for each time period
    rmse_all = []
    log_model_ens = []
    d_delta_u, d_delta_q = [], []
    for idx, (diagn, agg, masko) in enumerate(
            zip(cfg.predictor_diagnostics,
                cfg.predictor_aggs,
                cfg.predictor_masko)):
        logger.info('Calculate diagnostics for: {}'.format(diagn))

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
        log_model_ens.append([])
        for model_ens, filename in zip(*get_filenames(fn, varn, cfg.ensembles)):
            logger.debug('Calculate diagnostics for {}...'.format(model_ens))
            log_model_ens[-1].append(model_ens)

            # NOTE: I don't understand the special handling of 'tasclt'
            # if ((row['derived'] == True and diag == 'tasclt')):
            #     outfile = '%s%s%s' %(diagdir[diag], 'tas', file_start)
            # else:
            #     outfile = '%s%s' %(diagdir[diag], file_start)
            if cfg.predictor_derived[idx] and diagn == 'tasclt':
                filename_diag = calc_CORR(filename,
                                          base_path,
                                          varn, 'tas',
                                          masko=masko,
                                          syear=cfg.syear_eval[idx],
                                          eyear=cfg.eyear_eval[idx],
                                          season=cfg.predictor_seasons[idx],
                                          region=cfg.region)
            else:
                filename_template = os.path.join(base_path, os.path.basename(filename))
                filename_template = filename_template.replace('.nc', '')

                filename_diag = calc_diag(filename,
                                          filename_template,
                                          cfg.target_diagnostic,
                                          variable=varn, # ??
                                          masko=masko,
                                          syear=cfg.syear_eval[idx],
                                          eyear=cfg.eyear_eval[idx],
                                          season=cfg.target_season,
                                          kind=cfg.target_agg,
                                          region=cfg.region,
                                          overwrite=cfg.overwrite)

            # For each diagnostic, read data to calculate perfmetric
            fh = nc.Dataset(filename_diag, mode = 'r')
            try:
                diagnostic = fh.variables[diagn][:] # global data, time, lat, lon
            except KeyError:
                diagnostic = fh.variables[varn][:]
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
            fh.close()
            diagnostics.append(np.ma.filled(diagnostic, fill_value=np.nan))

        logger.info('Calculating perfmetric for all model combinations')
        rmse_models = np.empty((len(diagnostics), len(diagnostics)),
                            dtype=float) * np.nan

        for ii, diagnostic1 in enumerate(diagnostics):
            for jj, diagnostic2 in enumerate(diagnostics):
                if ii == jj:
                    rmse_models[ii, ii] = 0.0
                elif ii > jj:  # the matrix is symmetric
                    rmse_models[ii, jj] = rmse_models[jj, ii]
                else:
                    rmse_models[ii, jj] = rmse_weighted(diagnostic1, diagnostic2, lat)

        # read obs data if compared to obs and calculate perfmetric
        if cfg.obsdata is not None:
            logger.info('Reading obs data and calculating perfmetric')

            # TODO
            filename_obs = '%s%s_%s_%s_%s-%s_%s_%s_%s.nc' %(
                diagdir[diag], diag, cfg.freq, cfg.obsdata, cfg.syear_eval[i],
                cfg.eyear_eval[i], row['res_name'], row['var_file'], cfg.region)
            if not (os.path.isfile(filename_obs)):
                logger.error('Cannot find obs file %s, exiting.' %filename_obs)
                raise IOError

            fh = nc.Dataset(filename_obs, mode = 'r')
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
            try:
                obs = fh.variables[diagn][:] # global data,time,lat,lon
            except KeyError:
                obs = fh.variables[varn][:]
            fh.close()

            # mask model data where no obs
            # create mask based on obs
            mask_obs = np.ma.masked_invalid(obs).mask
            obs = np.ma.filled(obs, fill_value=np.nan)
            diagnostics = [np.ma.array(diag, mask=mask_obs).filled(np.nan)
                           for diag in diagnostics]

            rmse_obs = np.empty(len(diagnostics)) * np.nan
            for i_diag, diagnostic in enumerate(diagnostics):
                rmse_obs[i_diat] = rmse_weighted(diagnostic, obs, lat)


            rmse = np.concatenate((rmse_models, rmse_obs), axis=0)
        else:
            rmse = rmse_models

        rmse_all.append(rmse)

        # normalize deltas by median
        med = np.nanmedian(rmse)
        d_delta_u.append(np.divide(rmse_models, med))
        if cfg.obsdata:
            # NOTE: is this really the right way to normalize this??
            d_delta_q.append(np.divide(rmse_obs, med))
        # TODO: maybe we want to do this:
        # map the values of d_delta from [d_delta.min(), d_delta.max()] to [0, 1]
        # rmse_models = np.interp(rmse_models, [rmse_models.min(), rmse_models.max()], [0, 1])

    if cfg.obsdata:
        d_delta_u, d_delta_q = np.array(d_delta_u), np.array(d_delta_q)
        delta_u, delta_q = d_delta_u.mean(axis=0), d_delta_q.mean(axis=0)
    else:  # if there are not observations delta_u and delta_q are identical!
         delta_u = np.array(d_delta_u).mean(axis=0)
         delta_q = delta_u# .mean(axis=0)  # DEBUG: .mean(axis=0) is not in Ruths script

    if (log_model_ens[0] != log_model_ens[1] or
        log_model_ens[0] != log_model_ens[2]):
        raise ValueError

    return delta_q, delta_u, lat, lon

def calc_sigmas(targets, delta_u, lat, lon, fn, cfg, debug=False):
    """TODO: docsting"""

    targets = targets.squeeze()

    sigma_size = 41
    tmp = np.mean(delta_u)
    sigmas_q = np.linspace(.1*tmp, 1.9*tmp, sigma_size)
    sigmas_u = np.linspace(.1*tmp, 1.9*tmp, sigma_size)

    _, idx = np.unique(fn.indices['model'], return_index=True)  # index of unique models
    targets_1ens = targets[idx]
    delta_u_1ens = delta_u[idx, :][:, idx]
    targets_1ens_mean = utils.area_weighted_mean(targets_1ens, lat, lon)

    weights_sigmas, means_sigmas = calculate_weights_sigmas(
        targets_1ens_mean, delta_u_1ens, sigmas_q, sigmas_u)

    if debug:
        return weights_sigmas

    idx_u, idx_q = calculate_optimal_sigma(means_sigmas, targets_1ens_mean)

    import ipdb; ipdb.set_trace()

    # cfg.sigma_S2 =
    # cfg.sigma_D2 =


def calc_weights(delta_u, delta_q, target_data_ar, runs, cfg):
    logger.info('Calculate weights for model uniqueness (u) and quality (q)')
    wu_end = calc_wu(delta_u, cfg.sigma_S2)

    if cfg.obsdata:
        wq_end = calc_wq(delta_q, cfg.sigma_D2)

        # calculate weights and weighted multi-model mean
        weights, weighted_mmm = calc_weights_approx(wu_end, wq_end,
                                                    target_data_ar,
                                                    std = (cfg.target_file=='STD'))
        logger.info('Calculated weights are %s for models %s' %(
            str(weights), str(model_names)))
        sum_weights = np.sum(weights)
        logger.debug('Sum of all weights is %s' %(round(sum_weights, 3)))
    else:
        # do perfect model test for all models as truth if no obs given
        logger.info('Performing perfect model test and calculating weights')
        weights = dict()
        weighted_mmm = dict()
        for i, run in enumerate(runs):
            wq_end = calc_wq(delta_q[i, ], cfg.sigma_D2)
            # calculate weights and weighted multi-model mean
            tmp_weights, tmp_weighted_mmm = calc_weights_approx(wu_end,
                                                                wq_end,
                                                                target_data_ar,
                                                                std = (cfg.target_file=='STD'))
            weights[run] = tmp_weights # np.array shape[len(runs)]
            weighted_mmm[run] = tmp_weighted_mmm

    return weights, weighted_mmm


def save_data(weights, weighted_mmm, cfg):
    # put data into dict to save to json
    d_weights = dict()
    if cfg.obsdata:
        for mod in range(len(model_names)):
            d_weights[model_names[mod]] = weights[mod]
        d_weights['weighted multi-model-mean'] = weighted_mmm.tolist()
    else:
        for runname in model_names:
            d_weights1 = dict()
            for mod in range(len(model_names)):
                d_weights1[model_names[mod]] = weights[runname][mod]
            d_weights[runname] = d_weights1
            del d_weights1
            d_weights[runname]['weighted multi-model-mean'] = weighted_mmm[runname].tolist()

    # create string with all diagnostics names for output filename
    pred_str = ''
    for i, row in cfg.predictors.iterrows():
        tmp_str = ''.join([row['diag_var'], row['var_file'], '_'])
        pred_str += tmp_str

    # define path, filename and save data
    outdir_file = '%smodel_weights/%s/%s/%s/%s/%s/' %(
        cfg.outdir, cfg.target_var, cfg.target_file, cfg.target_mask, cfg.target_res, cfg.region)
    if (os.access(outdir_file, os.F_OK) == False):
        os.makedirs(outdir_file)

    outfile = '%s%s%s_sigmaS%s_sigmaD%s_%s_%s_%s-%s.txt' %(
        outdir_file, pred_str, len(cfg.predictors.diag_var), cfg.sigma_S2, cfg.sigma_D2,
        cfg.obsdata, cfg.region, cfg.syear_fut, cfg.eyear_fut)
    logger.info('Save data to json %s' %(outfile))
    with open(outfile, "w") as text_file:
        json.dump(d_weights, text_file)



def main():
    utils.set_logger(level = logging.INFO)
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
    delta_q, delta_u, lat, lon = calc_predictors(fn, cfg)
    logger.info('Calculate predictor diagnostics and delta matrix... DONE')

    if cfg.sigma_type == "inpercentile":
        logger.info('Calculate sigmas...')
        target_data_ar = calc_sigmas(targets, delta_u, lat, lon, fn, cfg)
        logger.info('Calculate sigmas... DONE')
    elif cfg.sigma_type == 'manual':
        logger.info('Using user sigmas: {}, {}'.format(cfg.sigma_S2, cfg.sigma_D2))
    else:
        errmsg = ' '.join(['simga_type has to be one of [interpercentile |',
                           'manual] not {}'.format(cfg.sigma_type)])
        logger.error(errmsg)
        raise NotImplementedError(errmsg)

    logger.info('Calculate weights and weighted mean...')
    calc_weights(delta_u, delta_q, target_data_ar, runs, cfg)
    logger.info('Calculate weights and weighted mean... DONE')

    logger.info('Saving data...')
    save_data(weights, weighted_mmm, cfg)
    logger.info('Saving data...')

    logger.info('Run program {}... Done'.format(os.path.basename(__file__)))


if __name__ == "__main__":
    main()
