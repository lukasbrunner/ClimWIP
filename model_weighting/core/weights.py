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
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def calculate_weights(quality, independence, sigma_q, sigma_i):
    """Calculates the (NOT normalised) weights for each model N.

    Parameters
    ----------
    quality : array_like, shape (N,)
        Array specifying the model quality.
    independence : array_like, shape (N, N)
        Array specifying the model independence.
    sigma_q : float
        Sigma value defining the form of the weighting function for the quality.
    sigma_i : float
        Sigma value defining the form of the weighting function for the independence.

    Returns
    -------
    numerator, denominator : ndarray, shape (N,)
    """
    assert len(quality.shape) == 1, 'quality needs to be a 1D array'
    assert len(independence.shape) == 2, 'independence needs to be a 2D array'
    errmsg = 'quality and independence need to have matching shapes'
    assert quality.shape == independence.shape[:1], errmsg
    assert isinstance(sigma_q, float), 'sigma_q needs to by of type float'
    assert isinstance(sigma_i, float), 'sigma_i needs to by of type float'
    assert np.all(np.isnan(np.diagonal(independence))), '(i, i) should be nan'
    assert np.isnan(quality).sum() <= 1, 'should have maximal one nan'

    numerator = np.exp(-((quality/sigma_q)**2))
    exp = np.exp(-((independence/sigma_i)**2))
    sum_exp = [np.sum(np.delete(ee, ii)) for ii, ee in enumerate(exp)]  # sum i!=j
    denominator = 1 + np.array(sum_exp)

    if sigma_i == -99.:
        denominator = (denominator * 0) + 1  # set to 1 (except NaN)
    if sigma_q == -99.:
        numerator = (numerator * 0) + 1  # set to 1 (except NaN)

    return numerator, denominator


def calculate_weights_sigmas(distances, sigmas_q, sigmas_i):
    """Calculates the weights for each model N and combination of sigma values.

    Parameters
    ----------
    distances : array_like, shape (N, N)
        Array specifying the distances between each model.
    sigmas_q : array_like, shape (M,)
        Array of sigma values for the weighting function of the quality.
    sigmas_i : array_like, shape (L,)
        Array of sigma values for the weighting function of the independence.

    Returns
    -------
    weights : ndarray, shape (M, L, N)
        Array of weights for each model and sigma combination.
    """
    ss = distances.shape
    assert len(ss) == 2, 'distances needs to be a 2D array'
    assert ss[0] == ss[1], 'distances needs to be of shape (N, N)'
    assert len(sigmas_q.shape) == 1, 'sigmas_q needs to be a 1D array'
    assert len(sigmas_i.shape) == 1, 'sigmas_i needs to be a 1D array'

    weights = np.zeros((len(sigmas_q), len(sigmas_i)) + distances.shape) * np.nan
    for idx_q, sigma_q in enumerate(sigmas_q):
        for idx_i, sigma_i in enumerate(sigmas_i):
            for idx_d, dd in enumerate(distances):
                # dd is the distance of each model to the idx_d-th model (='Truth')
                nu, de = calculate_weights(dd, distances, sigma_q, sigma_i)
                ww = nu/de
                assert np.isnan(ww[idx_d]), 'weight for model dd should be nan'
                ww[idx_d] = 0.  # set weight=0 to exclude the 'True' model
                assert ww.sum() != 0, 'weights = 0! sigma_q too small?'
                ww /= ww.sum()  # normalize weights
                # ww[ww < 1.e-10] = 0.  # set small weights to zero  # NOTE!!
                weights[idx_q, idx_i, idx_d] = ww
    return weights


def calculate_independence_ensembles(distances, sigmas_i):
    """Similar to calculate_weights_sigmas but only calculate the independence.

    Parameters:
    ----------
    distances : array_like, shape (N, N)
        Array specifying the distances between each model.
    sigmas_i : array_like, shape (M,)
        Array of sigma values for the weighting function of the independence.

    Returns
    -------
    independence_weights : ndarray, shape (M, N)
        Array of weights for each model and sigma combination.
    """
    ss = distances.shape
    assert len(ss) == 2, 'distances needs to be a 2D array'
    assert ss[0] == ss[1], 'distances needs to be of shape (N, N)'
    assert len(sigmas_i.shape) == 1, 'sigmas_i needs to be a 1D array'

    sigma_q = sigmas_i[0]  # dummy
    dd = distances[0]  # dummy
    weights = np.zeros((len(sigmas_i), len(dd))) * np.nan
    for idx_i, sigma_i in enumerate(sigmas_i):
        _, de = calculate_weights(dd, distances, sigma_q, sigma_i)
        weights[idx_i, :] = de
    return weights


def independence_sigma(delta_i, sigmas_i):
    """
    Estimate the independence sigma by using ensemble members.

    Parameters
    ----------
    delta_i : array_like, shape (N, N)
        Array specifying the distances between each model.
    sigmas_i : array_like, shape (M,)
        Array of sigma values for the weighting function of the independence.
    unique_models : list of strings
        A list with model identifiers to select only one member per model.

    Returns
    -------
    weighting_ratio : ndarray, shape (M,)
        Optimum is 1, i.e., models with x>1 members get correctly downweighted
        by a factor x and models with 1 member stay the same.
    """
    # get the number of initial-condition members per model
    model_ensemble = delta_i['model_ensemble'].data
    models = [*map(lambda x: x.split('_')[0], model_ensemble)]
    _, idx, counts = np.unique(models, return_index=True, return_counts=True)
    delta_i_1ens = delta_i.isel(model_ensemble=idx,
                                perfect_model_ensemble=idx).data
    indep_1ens = calculate_independence_ensembles(delta_i_1ens, sigmas_i)
    indep_ratio = []
    indep_ratio_others = []
    for jj, (ii, cc) in enumerate(zip(idx, counts)):
        if cc == 1:  # no ensembles
            continue
        # this index contains one member per model except for one model, for
        # which it contains all members (ii+1 because ii is already in idx)
        idx_1mod = np.sort(np.concatenate((idx, np.arange(ii+1, ii+cc))))
        delta_i_1mod = delta_i.data[idx_1mod, :][:, idx_1mod]
        indep_1mod = calculate_independence_ensembles(delta_i_1mod, sigmas_i)
        # calculate the mean weighting of all ensemble members
        temp = np.mean(indep_1mod[:, np.arange(jj, jj+cc)], axis=1)
        # remove 1 for each additional member and subtract the original weighting
        # -> in the ideal case this should be 0!
        temp = ((temp - (cc - 1)) - indep_1ens[:, jj])**2
        indep_ratio.append(temp)

        # calculate the mean weighting for all other models
        # -> should also be 0 in the ideal case (other models are not
        # influenced by adding ensemble members to one model).
        temp = np.delete(indep_1mod, np.arange(jj, jj+cc), axis=1)
        temp = np.mean((temp - np.delete(indep_1ens, jj, axis=1))**2, axis=1)
        indep_ratio_others.append(temp)

    indep_ratio_mean = np.mean(indep_ratio, axis=0)
    indep_ratio_others_mean = np.mean(indep_ratio_others, axis=0)
    weighting_ratio = np.sum([indep_ratio_mean, indep_ratio_others_mean], axis=0)
    idx_i_min = np.argmin(weighting_ratio)

    return np.atleast_1d(sigmas_i[idx_i_min])
