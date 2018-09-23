#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-23 18:52:32 lukas>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def calculate_weights(quality, independence, sigma_q, sigma_i):
    """Calculates the (not normalised) weights for each model N.

    Parameters:
    - quality (np.array): (N,) array specifying the model quality.
    - independence (np.array): (N, N) array specifying the model independence.
    - sigma_q (float): Sigma value defining the form of the weighting function
      for the quality.
    - sigma_i (float): Sigma value defining the form of the weighting function
      for the independence.

    Returns:
    weights (N,)"""
    assert len(quality.shape) == 1, 'quality needs to be a 1D array'
    assert len(independence.shape) == 2, 'quality needs to be a 2D array'
    errmsg = 'quality and independence need to have matching shapes'
    assert quality.shape == independence.shape[:1], errmsg
    assert isinstance(sigma_q, float), 'sigma_q needs to by of type float'
    assert isinstance(sigma_i, float), 'sigma_i needs to by of type float'

    numerator = np.exp(-((quality/sigma_q)**2))
    exp = np.exp(-((independence/sigma_i)**2))
    sum_exp = [np.sum(np.delete(ee, ii)) for ii, ee in enumerate(exp)]  # sum i!=j
    denominator = 1 + np.array(sum_exp)
    return numerator/denominator


def calculate_weights_sigmas(distances, sigmas_q, sigmas_i):
    """Calculates the weights for each model N and combination of sigma values.

    Parameters:
    - distances (np.array): (N, N) array specifying the distances between
      each model.
    - sigmas_q (np.array): (M,) array of possible sigma values for the
      weighting function of the quality.
    - sigmas_i (np.array): (L,) array of possible sigma values for the
      weighting function of the independence.

    Returns:
    weights (M, L, N)"""
    ss = distances.shape
    assert len(ss) == 2, 'distances needs to be a 2D array'
    assert ss[0] == ss[1], 'distances needs to be of shape (N, N)'
    assert len(sigmas_q.shape) == 1, 'sigmas_q needs to be a 1D array'
    assert len(sigmas_i.shape) == 1, 'sigmas_i needs to be a 1D array'

    weights = np.empty((len(sigmas_q), len(sigmas_i)) + distances.shape)*np.nan
    for idx_q, sigma_q in enumerate(sigmas_q):
        for idx_i, sigma_i in enumerate(sigmas_i):
            for idx_d, dd in enumerate(distances):
                # dd is the distance of each model to the idx_d-th model (='Truth')
                ww = calculate_weights(dd, distances, sigma_q, sigma_i)
                assert np.isnan(ww[idx_d]), 'weight for model dd should be nan'
                ww[idx_d] = 0.  # set weight=0 to exclude the 'True' model
                assert ww.sum() != 0, 'sigma_q chosen too small?'
                ww /= ww.sum()  # DEBUG: normalize just to be comparable with old script
                weights[idx_q, idx_i, idx_d] = ww
    return weights
