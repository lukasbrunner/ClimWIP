#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Lukas Brunner, ETH Zurich

This program is free software: you can redistribute it and/or modify
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
"""
import os
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_model_from_filename(filename):
    """Return the model identifier for a given filename."""
    parts = os.path.basename(filename).split('_')
    return f'{parts[2]}_{parts[4]}'


def get_filenames_var(varn, scenario, base_path):
    """Get all filenames matching the set criteria."""
    filename_pattern = f'{varn}_mon_*_{scenario}_*_g025.nc'
    fullpath = os.path.join(base_path, varn, filename_pattern)
    filenames = glob.glob(fullpath)
    return {get_model_from_filename(fn): fn for fn in filenames}


def get_unique_filenames(filenames, unique_models):
    """Return only one filename per model"""
    for varn in filenames.keys():
        filenames[varn] = {unique_model: filenames[varn][unique_model]
                           for unique_model in unique_models}
    return filenames


def get_filenames(varns, scenario, base_path, all_members):
    """
    Collects all filenames matching the set criteria.

    Collects all models (and initial-condition members) for each of the given
    variables. Then sub-selects models which are available for all given
    variables. Optionally sub-selects one initial-condition member per model.

    Parameters
    ----------
    varns : list of strings
        A list of valid CMIP5 variable names.
    scenario : string
        A valid CMIP5 scenario.
    base_path : string
        Base path of the CMIP5 next generation archive.
    all_members : bool
        If False only one initial-condition member per model will be used.
        Note that due to sorting this might not necessarily be the first
        (i.e., the r1i1p1) member (but rather, e.g., r10i1p1).

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
    filenames = {}
    for varn in varns:  # get all files for all variables first
        filenames[varn] = get_filenames_var(varn, scenario, base_path)
        try:
            common_models = np.intersect1d(common_models, list(filenames[varn].keys()))
        except NameError:
            common_models = list(filenames[varn].keys())

    unique_models = []
    unique_common_models = []
    for model_ensemble in common_models:
        model = model_ensemble.split('_')[0]
        if model not in unique_models:
            unique_models.append(model)
            unique_common_models.append(model_ensemble)

    for varn in varns:  # delete models not available for all variables
        delete_models = np.setdiff1d(list(filenames[varn].keys()), common_models)
        for delete_model in delete_models:
            filenames[varn].pop(delete_model)

    if not all_members:
        filenames = get_unique_filenames(filenames, unique_common_models)

    logger.info(f'{len(unique_common_models)} models found')
    logger.info(f'{len(filenames[varns[0]])} files found')
    logger.debug(f', '.join([model.split('_')[0] for model in unique_common_models]))
    logger.debug(', '.join(filenames[varns[0]].keys()))

    return filenames, np.array(unique_common_models)
