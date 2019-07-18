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
Performs a perfect model test. See 'perfect_model_test' docstring
for more information.
"""
import numpy as np

from .utils_xarray import quantile


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
    assert np.isclose(weights.sum(), 1., atol=.0001)
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
