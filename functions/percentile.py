#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-23 18:17:08 lukas>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import numpy as np


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    --> https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy#29677616
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def calculate_optimal_sigma(data, weights_sigmas, perc_lower=.1, perc_upper=.9):
    interpercentile = perc_upper - perc_lower  # DEBUG: comment in and delete next line
    # interpercentile = ((data > np.percentile(data, 10)) &
    #                    (data < np.percentile(data, 90))).sum() / float(len(data))
    inside_ratio = np.zeros(weights_sigmas.shape[:2])
    for ii in range(weights_sigmas.shape[0]):
        for jj in range(weights_sigmas.shape[1]):
            inside = np.zeros_like(data)
            for kk in range(len(data)):
                assert weights_sigmas[ii, jj, kk, kk] == 0  # CHECK
                pp1, pp2 = weighted_quantile(data, (perc_lower, perc_upper),
                                               weights_sigmas[ii, jj, kk])
                inside[kk] = (pp1 <= data[kk] <= pp2)
            inside_ratio[ii, jj] = inside.sum() / float(len(inside))
    inside_ok = inside_ratio >= interpercentile

    index_sum = 9999
    idx_i_min, idx_q_min = None, None
    for idx_q, qq in enumerate(inside_ok):
        if qq.sum() == 0:
            continue  # no fitting element
        elif idx_q >= index_sum:
            break  # no further optimization possible
        idx_i = np.where(qq==1)[0][0]
        if idx_i + idx_q < index_sum:
            index_sum = idx_i + idx_q
            idx_i_min, idx_q_min = idx_i, idx_q
    return idx_q_min, idx_i_min, inside_ratio
