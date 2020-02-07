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
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract
--------
Most file-path and filename dependent functions should be constrained here. If
you try to run ClimWIP with a different setup then the ETHZ next-generation
archive, you might need to change things here.
"""
import os
import glob
import logging
import warnings
import numpy as np
from natsort import natsorted, ns

logger = logging.getLogger(__name__)


def get_model_from_filename(filename, id_):
    """Return the model identifier for a given filename."""
    parts = os.path.basename(filename).split('_')
    return f'{parts[2]}_{parts[4]}_{id_}'


def cmip6_test_hist(filenames, scenario):
    """For CMIP6 we currently need to check if the historical file for a given
    scenario already exists as it is needed for merging in 'diagnostics.py'"""
    if scenario == 'historical':  # no need to check if scenario is historical
        return filenames

    del_files = []
    for filename in filenames:
        histfile = filename.replace(scenario, 'historical')
        if not os.path.isfile(histfile):
            del_files.append(filename)
    for del_file in del_files:
        filenames.remove(del_file)
    return filenames


def get_filenames_var(varn, id_, scenario, base_path):
    """Get all filenames matching the set criteria."""
    if id_ == 'CMIP6':
        filename_pattern = f'{varn}/mon/g025/{varn}_mon_*_{scenario}_*_g025.nc'
    elif id_ == 'CMIP5':
        filename_pattern = f'{varn}/{varn}_mon_*_{scenario}_*_g025.nc'
    elif id_ == 'CMIP3':
        filename_pattern = f'{varn}/{varn}_mon_*_{scenario}_*_g025.nc'
    elif id_ == 'LE':
        filename_pattern = f'{varn}_mon_*_{scenario}_*_g025.nc'
    else:
        raise ValueError(f'{id_} is not a valid model_id')
    fullpath = os.path.join(base_path, filename_pattern)
    filenames = glob.glob(fullpath)

    if id_ == 'CMIP6':
        filenames = cmip6_test_hist(filenames, scenario)

    assert len(filenames) != 0, f'no models found for {varn}, {id_}, {scenario}'
    return {get_model_from_filename(fn, id_): fn for fn in filenames}


def get_filenames_variants(filenames, unique_models):
    """Return only one filename per model"""
    for varn in filenames.keys():
        filenames[varn] = {unique_model: filenames[varn][unique_model]
                           for unique_model in unique_models}
    return filenames


def select_variants(common_model_ensembles, variants_use, variants_select):
    """
    Select the given number of variants of the same model (if avaliable).

    Parameters
    ----------
    common_model_ensembles : list of strings of form <model_ensemble_ID>
    variants_use : integer > 0 or 'all'
        The number of variants of the same model to use. This is an upper
        limit, if less variants are available all of them will be used
        (but they will not be repeated to reach the maximum number!).
        Setting this to 'all' has the same effect as setting it to a very
        high number (i.e., higher than the maximum number of variants for any
        given model).
    variants_select : {'natsorted', 'sorted', 'random'}
        The sorting strategy for model variants.
        * sorted: Sort using the Python buildin sorted() function. This was
          the original sorting strategy but leads to unexpected sorting:
          [r10i*, r11i*, r1i*, ...]
        * natsorted: Sort using the natsort.natsorted function:
          [r1i*, r10i*, r11i*, ...]
        * random: Do not sort but pick random members. This can be used for
          bootstrapping of model variants:
          [r24i*, r7i*, r13i*, ...]

    Returns
    -------
    selected_models : list of strings of from <model_ID>
        A list of unique models selected sorted by ID and model name.
    selected_model_ensembles : list of strings of from <model_ensemble_ID>
        A list of all models and model variants selected sorted by ID, model
        name, and variant.
    """
    if variants_use == 'all':
        variants_use = 999
    assert isinstance(variants_use, int) and variants_use > 0

    if variants_select == 'sorted':
        common_model_ensembles = sorted(common_model_ensembles)
    elif variants_select == 'natsorted':
        common_model_ensembles = natsorted(common_model_ensembles)
    elif variants_select == 'random':
        np.random.shuffle(common_model_ensembles)

    selected_models = []
    selected_model_ensembles = []
    for model_ensemble in common_model_ensembles:
        # extract model_ID (without variant information)
        model = model_ensemble.split('_')[0] + '_' + model_ensemble.split('_')[2]

        # check how many variants of the model are already selected (and add)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            nr_variants = sum([mm == model for mm in selected_models])
        if nr_variants < variants_use:
            selected_models.append(model)
            selected_model_ensembles.append(model_ensemble)

    return (natsorted(natsorted(np.unique(selected_models), alg=ns.IC), key=lambda x: x.split('_')[1]),
            natsorted(natsorted(selected_model_ensembles, alg=ns.IC), key=lambda x: x.split('_')[2]))


def get_filenames(cfg):
    """
    Collects all filenames matching the set criteria.

    Collects all models (and initial-condition members) for each of the given
    variables. Then sub-selects models which are available for all given
    variables. Optionally sub-selects one initial-condition member per model.

    Parameters
    ----------
    cfg : config object

    Returns
    -------
    filenames : nested dictionary
        A nested dictionary with the first layer having the given varns as keys
        and another dictionary as values. The second layer has model
        identifiers as keys and the filenames as values. E.g.,
        {'tas': {'model1_member1': 'path/filename.nc', ...}, ...}
    unique_common_models : list of strings
        A list of model identifiers to extract one initial-conditon member
        per model from filenames. Calling
        get_unique_filenames(filenames, unique_common_models) has the same
        effect as setting all_members=False. This is indented for use in the
        perfect model test.
    """
    # get basic variables for diagnostics
    varns = []
    if cfg.performance_diagnostics is not None:
        for varn in cfg.performance_diagnostics:
            if isinstance(varn, dict):  # multi-variable diagnostic
                varns.append([*varn.values()][0][0])
                varns.append([*varn.values()][0][1])
            else:
                varns.append(varn)
    if cfg.independence_diagnostics is not None:
        for varn in cfg.independence_diagnostics:
            if isinstance(varn, dict):
                varns.append([*varn.values()][0][0])
                varns.append([*varn.values()][0][1])
            else:
                varns.append(varn)
    if cfg.target_diagnostic is not None:
        # only if sigmas are None we need to calculate the target
        varns += [cfg.target_diagnostic]

    varns = np.unique(varns)  # we need each variable only once

    # common_model_ensembles: a list of model_ensemble_ID which are available for all variables
    # filenames: a nested list of filenames[varn][model_ensemble_ID] = filename
    # available of all variables
    filenames = {}
    for varn in varns:  # get all files for all variables first
        filenames[varn] = {}
        for id_, scenario, base_path in zip(cfg.model_id, cfg.model_scenario, cfg.model_path):
            filenames[varn].update(get_filenames_var(varn, id_, scenario, base_path))

        try:
            common_model_ensembles = list(
                np.intersect1d(common_model_ensembles, list(filenames[varn].keys())))
        except NameError:
            common_model_ensembles = list(filenames[varn].keys())

    for varn in varns:  # delete models not available for all variables
        delete_models = np.setdiff1d(list(filenames[varn].keys()), common_model_ensembles)
        for delete_model in delete_models:
            filenames[varn].pop(delete_model)

        if cfg.subset is not None:
            if not set(cfg.subset).issubset(list(filenames[varn].keys())):
                missing = set(cfg.subset).difference(list(filenames[varn].keys()))
                errmsg = ' '.join(['subset is not None but these models in',
                                   'subset were found for all variables:',
                                   ', '.join(missing)])
                raise ValueError(errmsg)
            delete_models = np.setdiff1d(list(filenames[varn].keys()), cfg.subset)
            for delete_model in delete_models:
                filenames[varn].pop(delete_model)

    if cfg.subset is not None:
        for delete_model in delete_models:
            common_model_ensembles.remove(delete_model)

    selected_models, selected_model_ensembles = select_variants(
        common_model_ensembles, cfg.variants_use, cfg.variants_select)

    if cfg.variants_use != 'all':
        filenames = get_filenames_variants(filenames, selected_model_ensembles)

    logger.info(f'{len(selected_models)} models found')
    logger.info(f'{len(selected_model_ensembles)} runs selected')
    logger.info(', '.join(selected_models))
    logger.info(', '.join(selected_model_ensembles))

    return filenames
