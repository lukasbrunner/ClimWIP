#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import logging
import numpy as np

from core import utils

logger = logging.getLogger(__name__)

types = {
    # --- other parameters ---
    'variants_use': (int, str),
    'variants_select': str,
    'variants_independence': bool,
    'idx_lats': (int, type(None)),
    'idx_lons': (int, type(None)),
    'inside_ratio': (float, str, type(None)),
    'overwrite': bool,
    'percentiles': float,
    'performance_metric': str,
    'plot_path': (str, type(None)),
    'plot': bool,
    'subset': (str, type(None)),
    'save_path': str,
    'sigma_i': (int, float, type(None)),
    'sigma_q': (int, float, type(None)),

    # --- data ---
    'model_path': str,
    'model_id': str,
    'model_scenario': str,
    'obs_path': str,
    'obs_id': str,
    'obs_uncertainty': (str, type(None)),

    # --- target ---
    'target_diagnostic': (str, type(None)),
    'target_agg': str,
    'target_season': str,
    'target_mask': (bool, str),
    'target_region': str,
    'target_startyear': int,
    'target_endyear': int,
    'target_startyear_ref': (int, type(None)),
    'target_endyear_ref': (int, type(None)),

    # --- performance ---
    'performance_diagnostics': (str, type(None)),
    'performance_aggs': (str, type(None)),
    'performance_seasons': (str, type(None)),
    'performance_masks': (bool, str, type(None)),
    'performance_regions': (str, type(None)),
    'performance_startyears': (int, type(None)),
    'performance_endyears': (int, type(None)),
    'performance_normalizers': (str, int, float, type(None)),
    'performance_weights': (int, float, type(None)),

    # --- independence ---
    'independence_diagnostics': str,
    'independence_aggs': str,
    'independence_seasons': str,
    'independence_masks': (bool, str),
    'independence_regions': str,
    'independence_startyears': int,
    'independence_endyears': int,
    'independence_normalizers': (str, int, float),
    'independence_weights': (int, float, type(None)),
}

values = {
    # --- other parameters ---
    'idx_lats': None,
    'idx_lons': None,
    'inside_ratio': None,  # TODO
    'overwrite': [True, False],
    'percentiles': None,
    'performance_metric': ['RMSE'],
    'plot_path': None,  # TODO: writable
    'plot': [True, False],
    'subset': None,
    'save_path': None,  # TODO: writable
    'sigma_i': None,
    'sigma_q': None,

    # --- data ---
    'model_path': None,  # TODO: exists
    'model_id': ['CMIP3', 'CMIP5', 'CMIP6', 'LE'],
    'model_scenario': None,  # TODO:
    'variants_use': ['all', *list(range(1, 100))],
    'variants_select': ['sorted', 'natsorted', 'random'],
    'variants_independence': [True, False],
    'obs_path': None,  # TODO: None or exists
    'obs_id': None,
    'obs_uncertainty': [None, 'range', 'mean', 'median', 'center'],

    # --- target ---
    'target_diagnostic': None,
    'target_agg': ['CLIM'],
    'target_season': ['ANN', 'JJA', 'SON', 'DJF', 'MAM'],
    'target_mask': [False, 'sea', 'land'],
    'target_region': None,   # TODO: check SREX or in folder
    'target_startyear': range(1850, 2101),
    'target_endyear': range(1850, 2101),
    'target_startyear_ref': range(1850, 2101),
    'target_endyear_ref': range(1850, 2101),

    # --- performance ---
    'performance_diagnostics': None,
    'performance_aggs': ['ANOM-GLOBAL', 'ANOM-LOCAL', 'CLIM', 'STD', 'TREND'],
    'performance_seasons': ['ANN', 'JJA', 'SON', 'DJF', 'MAM'],
    'performance_masks': [False, 'sea', 'land'],
    'performance_regions': None,  # TODO: check SREX or in folder
    'performance_startyears': range(1850, 2020),
    'performance_endyears': range(1850, 2020),
    'performance_normalizers': None,
    'performance_weights': None,

    # --- independence ---
    'independence_diagnostics': None,
    'independence_aggs': ['ANOM-GLOBAL', 'ANOM-LOCAL', 'CLIM', 'STD', 'TREND'],
    'independence_seasons': ['ANN', 'JJA', 'SON', 'DJF', 'MAM'],
    'independence_masks': [False, 'sea', 'land'],
    'independence_regions': None,  # TODO: check SREX or in folder
    'independence_startyears': range(1850, 2020),
    'independence_endyears': range(1850, 2020),
    'independence_normalizers': None,
    'independence_weights': None,
}


def check_types(cfg, types):
    for param in types.keys():
        if isinstance(cfg[param], list):
            for value in cfg[param]:
                if not isinstance(value, types[param]):
                    errmsg = ' '.join([
                        f'Wrong type in input field {param}! Each element',
                        f'should be {" or ".join(types[param])} at least one',
                        f'is {type(value)}!'])
                    logger.error(errmsg)
                    raise IOError(errmsg)
        else:
            if not isinstance(cfg[param], types[param]):
                errmsg = ' '.join([
                    f'Wrong type in input field {param}! Should be',
                    f'{", ".join(types[param])} is {type(value)}!'])
                logger.error(errmsg)
                raise IOError(errmsg)


def check_values(cfg, values):
    for param in values.keys():
        if values[param] is None:
            continue
        if isinstance(cfg[param], list):
            for value in cfg[param]:
                if value not in values[param]:
                    errmsg = ' '.join([
                        f'Wrong input value in field {param}! Each element',
                        f'should be in {", ".join(values[param])} at least one',
                        f'is not: {value}'])
                    logger.error(errmsg)
                    raise IOError(errmsg)
        else:
            if cfg[param] not in values[param]:
                errmsg = ' '.join([
                    f'Wrong input value in field {param}! Should be in',
                    f'{", ".join(values[param])} but is: {value}'])
                logger.error(errmsg)
                raise IOError(errmsg)


def set_default_values(cfg):
    try:
        cfg.subset
    except AttributeError:
        cfg.subset = None

    try:
        cfg.idx_lats
    except AttributeError:
        cfg.idx_lats = None

    try:
        cfg.idx_lons
    except AttributeError:
        cfg.idx_lons = None

    try:
        cfg.performance_metric
    except AttributeError:
        cfg.performance_metric = 'RMSE'

    independence_parameters = [
        'independence_diagnostics',
        'independence_aggs',
        'independence_seasons',
        'independence_masks',
        'independence_regions',
        'independence_startyears',
        'independence_endyears',
        'independence_normalizers',
        'independence_weights',
    ]
    # if non of them exists (old way) default to the same as performance
    if not np.all([hasattr(cfg, param) for param in independence_parameters]):
        for param in independence_parameters:
            cfg[param] = cfg[param.replace('independence_', 'performance_')]


def process_other_parameters(cfg):
    other_parameters = {
        'variants_use',
        'variants_select',
        'variants_independence',
        'idx_lats',
        'idx_lons',
        'inside_ratio',
        'overwrite',
        'percentiles',
        'performance_metric',
        'plot_path',
        'plot',
        'subset',
        'save_path',
    }

    for param in other_parameters:
        if param == 'variants_independence':
            if cfg['variants_independence'] and cfg['variants_use'] == 1:
                raise ValueError('Can not use variants_independence without variants')
        elif param == 'percentiles':
            if not np.all((value > 0) & (value < 1) for value in cfg[param]):
                raise ValueError
            if not len(cfg[param]) == 2:
                raise ValueError
        elif param == 'inside_ratio':
            if cfg[param] is None:
                cfg[param] = cfg['percentiles'][1] - cfg['percentiles'][0]
            elif cfg[param] == 'force':
                pass
            elif cfg[param] < 0 or cfg[param] > 1:
                raise ValueError

        elif param == 'performance_metric':
            if not cfg[param] in ['RMSE']:
                raise ValueError

        elif param == 'plot_path':
            if cfg['plot']:
                if cfg['plot_path'] is None:
                    raise ValueError
                if not os.access(cfg.plot_path, os.W_OK | os.X_OK):
                    raise ValueError('plot_path is not writable')

        elif param == 'save_path':
            if not os.access(cfg.plot_path, os.W_OK | os.X_OK):
                raise ValueError('save_path is not writable')


def process_model_parameters(cfg):
    model_parameters = {
        'model_path': str,
        'model_id': str,
        'model_scenario': str,
    }

    for param in model_parameters:
        if not isinstance(cfg[param], list):
            cfg[param] = [cfg[param]]
        if not np.all([isinstance(value, model_parameters[param])
                       for value in cfg[param]]):
            raise ValueError

    size = len(cfg.model_path)
    for param in model_parameters:
        if not len(cfg[param]) == size:
            raise ValueError


def process_obs_parameters(cfg):
    obs_parameters = {
        'obs_path': str,
        'obs_id': str,
        'obs_uncertainty': (str, type(None)),
    }

    if cfg.obs_path is None:
        for param in obs_parameters.keys():
            if cfg[param] is not None:
                raise ValueError
        return  # all parameters contain a single None

    param = 'obs_path'
    if cfg[param] is None:
        cfg['obs_id'] = None
    else:
        if not isinstance(cfg[param], list):
            cfg[param] = [cfg[param]]
            if not np.all([isinstance(value, obs_parameters[param])
                           for value in cfg[param]]):
                raise ValueError
        size = len(cfg[param])

    param = 'obs_id'
    if cfg[param] is None:
        cfg['obs_path'] = None
    else:
        if not isinstance(cfg[param], list):
            cfg[param] = [cfg[param]]
        if not np.all([isinstance(value, obs_parameters[param])
                       for value in cfg[param]]):
            raise ValueError
        if not len(cfg[param]) == size:
            raise ValueError

    param = 'obs_uncertainty'
    if not isinstance(cfg[param], obs_parameters[param]):
        raise ValueError
    if cfg[param] is None and not (len(cfg['obs_path']) == 1 or cfg['obs_path'] is None):
        raise ValueError


def process_target_parameters(cfg):
    target_parameters = {
        'target_diagnostic': str,
        'target_agg': str,
        'target_season': str,
        'target_mask': (bool, str),
        'target_region': str,
        'target_startyear': int,
        'target_endyear': int,
        'target_startyear_ref': (int, type(None)),
        'target_endyear_ref': (int, type(None)),
    }
    if cfg.target_diagnostic is None:
        # for param in target_parameters.keys():
        #     if cfg[param] is not None:
        #         raise ValueError
        return  # all parameters contain a single None

    for param in target_parameters.keys():
        if not isinstance(cfg[param], target_parameters[param]):
            raise ValueError


def process_multi_vars(cfg):
    for param in ['performance_diagnostics', 'independence_diagnostics']:
        if cfg[param] is not None:
            for idx, diagn in enumerate(cfg[param]):
                if '-' in diagn:
                    varns = diagn.split('-')
                    assert len(varns) == 2
                    cfg[param][idx] = {diagn: varns}


def process_performance_parameters(cfg):
    performance_parameters = {
        'performance_diagnostics': str,
        'performance_aggs': str,
        'performance_seasons': str,
        'performance_masks': (bool, str),
        'performance_regions': str,
        'performance_startyears': int,
        'performance_endyears': int,
        'performance_normalizers': (str, int, float),
        'performance_weights': (int, float, type(None)),
    }

    expand_parameters = [
        'performance_seasons',
        'performance_masks',
        'performance_regions',
        'performance_startyears',
        'performance_endyears',
        'performance_normalizers',
        'performance_weights',
    ]

    if cfg.performance_diagnostics is None:  # no performance weighting
        for param in performance_parameters.keys():
            if cfg[param] is not None:
                raise ValueError
        return  # all parameters contain a single None

    if not isinstance(cfg.performance_diagnostics, list):  # only one diagnostic
        for param in performance_parameters.keys():
            if isinstance(cfg[param], list):  # then all need to be not a list
                raise ValueError
            else:
                cfg[param] = [cfg[param]]

    size = len(cfg.performance_diagnostics)
    for param in expand_parameters:
        if not isinstance(cfg[param], list) and cfg[param] is not None:
            # allow only one value for some parameters and expand to list here
            cfg[param] = [cfg[param]] * size

    for param in performance_parameters.keys():
        if cfg[param] is None:
            continue
        if not isinstance(cfg[param], list):
            raise ValueError
        if len(cfg[param]) != size:
            errmsg = ' '.join([
                'All performance_* parameters need to have same length',
                f'{param} has {len(cfg[param])} but should have {size}'])
            logger.error(errmsg)
            raise ValueError(errmsg)
        if not np.all([isinstance(value, performance_parameters[param])
                       for value in cfg[param]]):
            raise ValueError


def process_independence_parameters(cfg):
    independence_parameters = {
        'independence_diagnostics': str,
        'independence_aggs': str,
        'independence_seasons': str,
        'independence_masks': (bool, str),
        'independence_regions': str,
        'independence_startyears': int,
        'independence_endyears': int,
        'independence_normalizers': (str, int, float),
        'independence_weights': (int, float, type(None)),
    }

    expand_parameters = [
        'independence_seasons',
        'independence_masks',
        'independence_regions',
        'independence_startyears',
        'independence_endyears',
        'independence_normalizers',
        'independence_weights',
    ]

    if cfg.independence_diagnostics is None:  # no independence weighting
        for param in independence_parameters.keys():
            if cfg[param] is not None:
                raise ValueError
        return  # all parameters contain a single None

    if not isinstance(cfg.independence_diagnostics, list):  # only one diagnostic
        for param in independence_parameters.keys():
            if isinstance(cfg[param], list):
                raise ValueError
            else:
                cfg[param] = [cfg[param]]

    size = len(cfg.independence_diagnostics)
    for param in expand_parameters:
        if not isinstance(cfg[param], list) and cfg[param] is not None:
            # allow only one value for some parameters and expand to list here
            cfg[param] = [cfg[param]] * size

    for param in independence_parameters.keys():
        if cfg[param] is None:
            continue
        if not isinstance(cfg[param], list):
            raise ValueError
        if len(cfg[param]) != size:
            raise ValueError
        if not np.all([isinstance(value, independence_parameters[param])
                       for value in cfg[param]]):
            raise ValueError


def process_sigmas(cfg):
    if cfg.target_diagnostic is None:
        if cfg.sigma_i is None or cfg.sigma_q is None:
            errmsg = 'If target_diagnostic is None, both sigmas need to be set!'
            raise ValueError(errmsg)

    # TODO: is it allowed to have one sigma None and the other set?
    # TODO: I think I can remove the sigma = -99 case with the new separation between
    # dependence and performance


def check_perfect_model_test(cfg):
    """
    For the perfect model test the performance and independence parameters
    need to be identical. So setting them separately is only allowed if the
    sigma values are use-given so that the perfect model test can be omitted.
    """
    parameters = [
        '{}_diagnostics',
        '{}_aggs',
        '{}_seasons',
        '{}_masks',
        '{}_regions',
        '{}_startyears',
        '{}_endyears',
        '{}_normalizers',
        '{}_weights',
    ]
    same = True
    for param in parameters:
        if ((cfg[param.format('performance')] is None and cfg[param.format('independence')] is not None) or
            (cfg[param.format('performance')] is not None and cfg[param.format('independence')] is None)):
            same = False
        elif cfg[param.format('performance')] is None and cfg[param.format('independence')] is None:
            pass
        elif not tuple(cfg[param.format('performance')]) == tuple(cfg[param.format('independence')]):
            same = False

    if not same and (cfg.sigma_i is None or cfg.sigma_q is None):
        errmsg = 'If performance_* and independence_* parameters are not identical sigmas have to be set!'
        raise ValueError(errmsg)

    if not same and cfg.obs_id is None:
        errmsg = 'If performance_* and independence_* parameters are not identical obs_id has to be set!'
        raise ValueError(errmsg)


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
    set_default_values(cfg)
    process_other_parameters(cfg)
    process_model_parameters(cfg)
    process_obs_parameters(cfg)
    process_target_parameters(cfg)
    process_performance_parameters(cfg)
    process_independence_parameters(cfg)
    process_multi_vars(cfg)
    process_sigmas(cfg)
    check_perfect_model_test(cfg)
    utils.log_parser(cfg)
    return cfg
