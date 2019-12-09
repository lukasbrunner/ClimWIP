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
import numpy as np

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


def get_unique_filenames(filenames, unique_models):
    """Return only one filename per model"""
    for varn in filenames.keys():
        filenames[varn] = {unique_model: filenames[varn][unique_model]
                           for unique_model in unique_models}
    return filenames


def get_filenames(varns, ids, scenarios, base_paths, all_members, subset=None):
    """
    Collects all filenames matching the set criteria.

    Collects all models (and initial-condition members) for each of the given
    variables. Then sub-selects models which are available for all given
    variables. Optionally sub-selects one initial-condition member per model.

    Parameters
    ----------
    varns : list of strings
        A list of valid CMIP5 variable names.
    ids : string or list of strings {'CMIP6', 'CMIP5', 'CMIP3'}
        Valid model IDs.
    scenarios : string or list of strings
        Valid scenarios. Must have same length as ids.
    base_paths : string or list of strings
        Base paths of the model archives. Must have same length as ids.
    all_members : bool
        If False only one initial-condition member per model will be used.
        Note that due to sorting this might not necessarily be the first
        (i.e., the r1i1p1) member (but rather, e.g., r10i1p1).
    subset : None or list of strings
        If not None must be a list of valid model identifyers (e.g.,
        ['CESM12_r1i1pi_CMIP5']). If one of the models available for all
        variables a ValueError will be raised to ensure that all models in the
        list are included.

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

    if isinstance(ids, str):
        ids = [ids]
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    if len(ids) != len(scenarios) or len(ids) != len(base_paths):
        raise ValueError('ids, scenarios, and base_paths need to have same lenght')
    filenames = {}
    for varn in varns:  # get all files for all variables first
        filenames[varn] = {}
        for id_, scenario, base_path in zip(ids, scenarios, base_paths):
            filenames[varn].update(get_filenames_var(varn, id_, scenario, base_path))

        try:
            common_models = list(
                np.intersect1d(common_models, list(filenames[varn].keys())))
        except NameError:
            common_models = list(filenames[varn].keys())

    for varn in varns:  # delete models not available for all variables
        delete_models = np.setdiff1d(list(filenames[varn].keys()), common_models)
        for delete_model in delete_models:
            filenames[varn].pop(delete_model)

        if subset is not None:
            if not set(subset).issubset(list(filenames[varn].keys())):
                missing = set(subset).difference(list(filenames[varn].keys()))
                errmsg = ' '.join(['subset is not None but these models in',
                                   'subset were found for all variables:',
                                   ', '.join(missing)])
                raise ValueError(errmsg)
            delete_models = np.setdiff1d(list(filenames[varn].keys()), subset)
            for delete_model in delete_models:
                filenames[varn].pop(delete_model)

    if subset is not None:
        for delete_model in delete_models:
            common_models.remove(delete_model)

    unique_models = []
    unique_common_models = []
    for model_ensemble in common_models:
        model = model_ensemble.split('_')[0] + '_' + model_ensemble.split('_')[2]
        if model not in unique_models:
            unique_models.append(model)
            unique_common_models.append(model_ensemble)

    if not all_members:
        filenames = get_unique_filenames(filenames, unique_common_models)

    logger.info(f'{len(unique_common_models)} models found')
    logger.info(f'{len(filenames[varns[0]])} runs selected')
    logger.info(f', '.join(sorted(unique_models, key=lambda x: x.split('_')[1])))
    logger.info(', '.join(filenames[varns[0]].keys()))

    return filenames, np.array(unique_common_models)
