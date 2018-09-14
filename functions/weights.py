#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-06-05 11:13:31 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Ruth Lorenz || ruth.lorenz@env.ethz.ch
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import numpy as np


def calc_wu(delta, sigma=.6):
    """
    Calculate weights accounting for dependence between models.

    Parameters:
    - delta (np.array): Shape (N,N), first dimension has to be 'models'
    - sigma (list of float): Shape (M,)

    Returns:
    - np.array shape=(M,N) (squeezed if M=1)
    """
    if isinstance(sigma, float):
        sigma = [sigma]
    wu = np.empty((len(sigma), delta.shape[0])) * np.nan
    for i_sig, sig in enumerate(sigma):
        S = np.exp(-((delta/sig)**2))  # (N, N)
        wu[i_sig] = [1. / (1 + np.sum(np.delete(ss, ii))) # (N, N)
                           for ii, ss in enumerate(S)]  # sum over i!=j, (N)

    return wu.squeeze()


def calc_wq(delta, sigma=.6):
    """
    Calculate weights accounting for model quality.

    Parameters:
    - delta (np.array): Shape (N,...), first dimension has to be 'models'
    - sigma (list of float): Shape (M,)

    Returns:
    - np.array shape=(M,N) (squeezed if M=1)
    """
    if isinstance(sigma, float):
        sigma = [sigma]
    wq = np.empty((len(sigma),) + np.shape(delta)) * np.nan
    for i_sig, sig in enumerate(sigma):
        wq[i_sig] = np.exp(-((delta/sig)**2))

    return wq.squeeze()


def calc_weights_approx(wu, wq, data, std=False):
    """
    Calculate weights and approximation based on weights (weighted multi-model
    mean)

    Parameters:
    - wu (np.array): Shape (M,)
    - wq (np.array): Shape (M,)
    - data (np.array): Shape (N,...)
    - std=False (bool, optional): If True the diagnostic is based on the
    standard deviation, therefor the multi model mean is a mean of std which
    cannot simply be calculated using mean

    Returns:
    - tuple of two np.array, weights has shape (N), approx same shape as data
    """
    def multiply_along_axis(arr, arr1D, axis):
    # https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
        dim_array = np.ones((1, arr.ndim), int).ravel()
        dim_array[axis] = -1
        arr1D_resh = arr1D.reshape(dim_array)
        return arr * arr1D_resh

    w_prod = np.empty((len(wq))) * np.nan
    for j in range(len(wq)):
        tmp_wu = wu[j]
        w_prod[j] = wq[j] * tmp_wu

    wu_wq_sum = np.nansum(w_prod)

    if wu_wq_sum != 0.:
        weights =  w_prod / wu_wq_sum
    else:  # how can this ever happen? if wq and wu are both very small (and numerically rounded to zero)
        weights = w_prod * 0.0

    if std:
        approx = np.sqrt(
            np.sum(multiply_along_axis(data**2, weights, axis = 0), axis = 0) /
            np.nansum(weights))
    else:
        approx = (np.sum(multiply_along_axis(data, weights, axis = 0), axis = 0)
            / np.nansum(weights))

    return weights, approx


def calc_weights_approx_multiple_sigmas(wu, wq, data, std = False):
    """
    Calculate weights and approximation based on weights (weighted multi-model
    mean) for multiple sigma combinations, weights have shape (S1, S2, N)
    and approx as well.

    Parameters:
    - wu, wq (np.array): Shape (S1, S2, N)
    - data (np.array): Shape (N,...)
    - std=False (bool, optional): If True the diagnostic is based on the
    standard deviation, therefor the multi model mean is a mean of std which
    cannot simply be calculated using mean

    Returns:
    - tuple of two np.array, weights has shape (S1, S2, N),
      approx shape (S1, S2, N, ...) (... same as data)
    """

    if len(wq) == data.shape[0]:
        # could be done with calc_weights_approx only
        weights, approx = calc_weights_approx(wu, wq, data, std)
    else:
        approx = np.empty((len(wu), len(wq),) + np.shape(data)) * np.nan
        # for each sigma combination, for each model as truth 1weight per model 
        weights = np.empty((len(wu), len(wq), data.shape[0],
                            data.shape[0])) * np.nan

        for u in range(len(wu)):
            for q in range(len(wq)):
                for j in range(data.shape[0]):
                    tmp_wu = wu[u, :].copy()
                    tmp_wq = wq[q, j, :]
                    tmp_wu[j] = 0.0
                    tmp_wq[j] = 0.0
                    #import ipdb ; ipdb.set_trace()
                    weights[u, q, j, :], approx[u, q, j, :] = calc_weights_approx(tmp_wu, tmp_wq, data, std)

    return weights, approx  #, std
