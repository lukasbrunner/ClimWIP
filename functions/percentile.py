#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2018-09-19 15:21:51 lukbrunn>

(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import numpy as np


def calculate_optimal_sigma(means_sigmas, truths):
    inside_80 = np.empty(means_sigmas.shape[:2]) * np.nan
    ratios = np.empty(means_sigmas.shape[:2]) * np.nan  # DEBUG
    for ii in range(means_sigmas.shape[0]):
        for jj in range(means_sigmas.shape[1]):
            p05 = np.percentile(means_sigmas[ii, jj], 5, interpolation='lower')
            p95 = np.percentile(means_sigmas[ii, jj], 95, interpolation='higher')
            # p05, p95 = np.percentile(means_sigmas[ii, jj], (5, 95))
            inside = np.where((truths >= p05) & (truths <= p95), 1, 0)
            ratios[ii, jj] = inside.sum() / float(len(inside))  # DEBUG
            inside_80[ii, jj] = (inside.sum() / float(len(inside))) >= .75  # DEBUG

    index_sum = 999
    idx_i_min, idx_q_min = None, None
    for idx_i, ii in enumerate(inside_80):
        if ii.sum() == 0:
            continue  # save time: no fitting element
        elif idx_i > index_sum:
            break  # save time: no further optimization possible
        _, idx_q = np.unique(ii, return_index=True)
        if idx_i + idx_q < index_sum:
            index_sum = idx_i + idx_q
            idx_i_min, idx_q_min = idx_i, idx_q

    return idx_i_min, idx_q_min
