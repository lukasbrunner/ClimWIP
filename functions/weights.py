#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-19 12:57:53 lukbrunn>

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
    """Calculates the weights for each model N.

    Parameters:
    - quality (np.array): (N,) array specifying the model quality. TODO: need to be normalized?
    - independence (np.array): (N, N) array specifying the model independence. # TODO: need to be normalized?
    - sigma_q (float): Sigma value defining the form of the weighting function
      for the quality.
    - sigma_i (float): Sigma value defining the form of the weighting function
      for the independence.

    Returns:
    np.array (N,)"""
    if len(quality.shape) != 1 or len(independence.shape) != 2:
        errmsg = 'quality and independence need to be 1D and 2D arrays, respectively'
        logger.error(errmsg)
        raise IOError(errmsg)
    if quality.shape != independence.shape[:1]:
        errmsg = 'quality and independence need have the same length'
        logger.error(errmsg)
        raise IOError(errmsg)
    numerator = np.exp(-((quality/sigma_q)**2))

    exp = np.exp(-((independence/sigma_i)**2))
    sum_exp = [np.sum(np.delete(ee, ii)) for ii, ee in enumerate(exp)]  # sum i!=j
    denominator = 1 + np.array(sum_exp)
    return numerator/denominator


def calculate_weights_sigmas(data, distances, sigmas_q, sigmas_i):
    """Calculates the weights for each model N and combination of sigma values.
    Also calculates the weighted model mean for each combination on the fly.

    Parameters:
    - data (np.array): (N,...) array of model data
    - distances (np.array): (N, N) array specifying the distances between
      each model.
    - sigmas_q (np.array): (M,) array of possible sigma values for the
      weighting function of the quality.
    - sigmas_i (np.array): (L,) array of possible sigma values for the
      weighting function of the independence.

    Returns:
    weights (M, L, N), weighted_mean (M, L,...)"""
    if len(distances.shape) != 2:
        errmsg = 'distances needs to be a 2D array'
        logger.error(errmsg)
        raise IOError(errmsg)
    if distances.shape[0] != distances.shape[1]:
        errmsg = 'distances needs to be of shape (N, N)'
        logger.error(errmsg)
        raise IOError(errmsg)
    if data.shape[0] != distances.shape[0]:
        errmsg = 'First dimensions of data and distances need to match'
        logger.error(errmsg)
        raise IOError(errmsg)

    weights = np.empty((len(sigmas_q), len(sigmas_i)) + distances.shape) * np.nan
    mean = np.empty((len(sigmas_q), len(sigmas_i), distances.shape[0])) * np.nan
    for idx_q, sigma_q in enumerate(sigmas_q):
        for idx_i, sigma_i in enumerate(sigmas_i):
            for idx_d, dd in enumerate(distances):
                # dd is the distance of each model to the idx_d-th model (='Truth')
                ww = calculate_weights(dd, distances, sigma_q, sigma_i)
                # exclude the 'True' model by setting the weight to zero
                ww[idx_d] = 0.
                ww /= ww.sum()  # DEBUG: normalize just to be comparable with old script
                mean[idx_q, idx_i, idx_d] = np.average(data, weights=ww, axis=0)
                weights[idx_q, idx_i, idx_d] = ww

    return weights, mean
