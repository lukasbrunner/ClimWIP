#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-06-27 13:49:17 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract: Performs a perfect model test. See 'perfect_model_test' docstring
for more information."""
import numpy as np


def quantile(data, quantiles, weights=None, interpolation='linear',
             old_style=False):
    """Calculates weighted quantiles.

    Parameters:
    - data (np.array): Array of data (N,)
    - quantiles (np.array): Array of quantiles (M,) in [0, 1]
    - weights=None (np.array, optional): Array of weights (N,)
    - interpolation='linear' (str, optional): String giving the interpolation
      method (equivalent to np.percentile). "This optional parameter specifies
      the interpolation method to use when the desired quantile lies between
      two data points." One of (with i < j):
      * linear: i + (j - i) * fraction where fraction is the fractional part
        of the index surrounded by i and j
      * lower: i  NOTE: might lead to unexpected results for integers (see
        tests/test_math.test_quantile_interpolation)
      * higher: j  NOTE: might lead to unexpected results for integers
      * nearest: i or j whichever is nearest
      * midpoint: (i + j) / 2. TODO: not yet implemented!
    - old_style=False (bool, optional): If True, will correct output to be
      consistent with np.percentile.

    Returns:
    np.array of shape (M,)"""
    data = np.array(data)
    quantiles = np.array(quantiles)
    if np.any(np.isnan(data)):
        errmsg = ' '.join([
            'This function is not tested with missing data! Comment this test',
            'if you want to use it anyway.'])
        raise ValueError(errmsg)
    if data.ndim != 1:
        errmsg = 'data should have shape (N,) not {}'.format(data.shape)
        raise ValueError(errmsg)
    if np.any(quantiles < 0.) or np.any(quantiles > 1.):
        errmsg = 'quantiles should be in [0, 1] not {}'.format(quantiles)
        raise ValueError(errmsg)
    if weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.array(weights)
        if data.shape != weights.shape:
            errmsg = ' '.join([
                'weights need to have the same shape as data ',
                '({} != {})'.format(weights.shape, data.shape)])
            raise ValueError(errmsg)
        # remove values with weights zero
        idx = np.where(weights == 0)[0]
        weights = np.delete(weights, idx)
        data = np.delete(data, idx)

    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - .5*weights

    if old_style:  # consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:  # more correct (see reference for a discussion)
        weighted_quantiles /= np.sum(weights)

    results = np.interp(quantiles, weighted_quantiles, data)

    if interpolation == 'linear':
        return results
    elif interpolation == 'lower':
        if isinstance(results, float):
            return data[data<=results][-1]
        return np.array([data[data<=rr][-1] for rr in results])
    elif interpolation == 'higher':
        if isinstance(results, float):
            return data[data>=results][0]
        return np.array([data[data>=rr][0] for rr in results])
    elif interpolation == 'nearest':
        if isinstance(results, float):
            return data[np.argmin(np.abs(data - results))]
        return np.array([data[np.argmin(np.abs(data - rr))] for rr in results])
    elif interpolation == 'midpoint':
        raise NotImplementedError
    else:
        errmsg = ' '.join([
            'interpolation has to be one of [linear | lower | higher |',
            'nearest | midpoint] and not {}'.format(interpolation)])
        raise ValueError(errmsg)


def weighted_quantile(values, weights, quantiles):
    """Very close to numpy.percentile, but supports weights.

    NOTE: this is a vectorized version in the 'weights' parameter.
    It is written and tested for weights of shape (L, L, N, N), where
    L is the length of both sigma_q & sigma_i and N is the number of models.
    See 'Special requirements' for more information.

    Parameters
    ----------
    values : array_like, shape (N,)
        Array of values.
    weights : array_like, shape (..., N)
        Array of values, last dimension has to match values.
    quantiles : array_like, shape(2,)
        Array of quantiles, has to be of length 2.

    Returns
    -------
    quantiles : ndarray, shape (..., 2)
        Array of quantiles of same shape as weights except for the last
        dimension which will contain the lower and upper quantile.

    Special requirements
    --------------------
    - the last dimension of 'weights' needs to be normalized
    - quantiles needs to have len = 2 (upper and lower quantile)
    - one weight should be zero (perfect model test)

    """
    values = np.array(values)
    weights = np.array(weights)
    quantiles = np.array(quantiles)
    assert np.isclose(weights.sum(), 1., atol=.00001)
    assert quantiles.size == 2
    assert quantiles[0] < quantiles[1]
    assert len(weights[weights < 1.e-10]) >= 1, 'at least perfect model weight should be 0'
    return quantile(values, quantiles, weights=weights)


weighted_quantile = np.vectorize(weighted_quantile, excluded=[0, 2], signature='(n)->(m)')


def perfect_model_test(data, weights_sigmas, perc_lower, perc_upper):
    """Perform a perfect model test.

    Parameters
    ----------
    data : array_like, shape (M,)
        Array of data.
    weights_sigmas : array_like, shape (N, N, M, M)
        Array of weights
    perc_lower : float
        Has to be in [0, 1] and < perc_upper
    perc_upper : float
        Has to be in [0, 1] and > perc_lower

    Returns
    -------
    inside_ratio : ndarray, shape(N, N)
        See Information

    Information
    -----------
    - data represents m=0...M models.
    - weights_sigmas represents NxN different sigma combinations as well as
      M weights for with each of the M models as 'truth' once (MxM combinations)

    For a given sigma combination n in NxN and a model m in M:
    data = [d1, d2,..., dm, ..., dM]  where 'dm' is the data of 'true' model m
    weights_sigmas = [w1, w2,... wm, ..., wM]  where wm=0 is the weight of the 'true' model m

    - wm is zero by default since the true model is excluded from the computation of percentiles
    - from all other models the upper and lower weighted percentiles are calculated
    - it is tested if the 'true' model m lies within these percentiles
      (for the unweighted case we would expect given member of a distribution to lay within
      two given percentiles in (perc_upper - perc_lower) of cases (i.e., 80% of the time for
      0.1 and 0.9 as percentiles))
    - this is repeated for each of the M models (and corresponding weights) as 'truth'
    - the inside ratio for a given sigma combination n is the ratio of the number of
      'true' models lying within their weighted percentiles to the total number
      of models.
    """
    tmp = weighted_quantile(data, weights_sigmas, (perc_lower, perc_upper))
    assert np.all(tmp[..., 0] <= tmp[..., 1])
    errmsg = 'Lower and upper percentile equivalent! Too strong weighting?'
    assert not np.any(np.isclose(tmp[..., 1] - tmp[..., 0], 0, atol=1.e-5)), errmsg
    inside = (tmp[..., 0] <= data) & (data <= tmp[..., 1])
    return inside.sum(axis=-1) / float(inside.shape[-1])
