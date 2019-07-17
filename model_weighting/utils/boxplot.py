#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-07-17 18:44:38 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


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


def boxplot(ax,
            pos=None,
            data=None,
            median=None,
            mean=None,
            box=None,
            whis=None,
            weights=None,
            width=.8,
            color=sns.xkcd_rgb['greyish'],
            alpha=1.,
            showcaps=True,
            fancy_legend=False,
            return_handle=False,
            zorder=100,
            median_kwargs=None,
            mean_kwargs=None,
            whis_kwargs=None):
    """
    A custom-mad boxplot routine based on user set statistics.

    Parameters
    ----------
    ax : plt.axis object
    pos : float, optional
        Location (center) of the box.
    data: array-like, optional
        An array of data to calculate the statistics from. If this is
        not None, median, mean, box, and whis will be overwritten.
    median : float or array-like, optional
        Location of the median or data to calculate the median.
    mean : float or array-like, optional
        Location of the mean or data to calculate the mean.
    box : tuple of float or array-like, shape (2), optional
        Extend of the box or data to calculate the box.
    whis : tuple of float or array-like, shape (2), optional
        Extend of the whiskers or data to calculate the whiskers.
    width : float, optional
        Width of box, median, mean, and caps (caps have .4*width).
    color : string, optional
        Box color and default color for median, mean, and whiskers.
    showcaps : bool, optional
        Whether to draw caps at the end of the whiskers.
    zorder : int, optional
        zorder of the drawn objects.
    median_kwargs : dict, optional
        Keyword arguments passed on to ax.hlines for the median.
    mean_kwargs : dict, optional
        Keyword arguments passed on to ax.hlines for the mean.
    whis_kwargs : dict, optional
        Keyword arguments passed on to ax.hlines and ax.vlines for whiskers
        and caps.
    """
    if data is not None:
        mean = data
        median = data
        box = data
        whis = data

    if mean is not None and not isinstance(mean, (int, float)):
        mean = np.average(mean, weights=weights)
    if median is not None and not isinstance(median, (int, float)):
        median = quantile(median, .5, weights)
    if box is not None and len(box) != 2 and not isinstance(box, tuple):
        box = quantile(box, (.25, .75), weights)
    elif box == (None, None):
        box = None
    if whis is not None and len(whis) != 2 and not isinstance(whis, tuple):
        whis = quantile(whis, (.05, .95), weights)
    elif whis == (None, None):
        whis = None

    if pos is None:
        pos = 0
    if median_kwargs is None:
        median_kwargs = {}
    if mean_kwargs is None:
        mean_kwargs = {}
    if whis_kwargs is None:
        whis_kwargs = {}
    if 'colors' not in median_kwargs.keys():
        median_kwargs['colors'] = 'k'
    if 'colors' not in mean_kwargs.keys():
        mean_kwargs['colors'] = 'k'
    if 'colors' not in whis_kwargs.keys():
        whis_kwargs['colors'] = color
    if 'alpha' not in median_kwargs.keys():
        median_kwargs['alpha'] = alpha
    if 'alpha' not in mean_kwargs.keys():
        mean_kwargs['alpha'] = alpha
    if 'alpha' not in whis_kwargs.keys():
        whis_kwargs['alpha'] = alpha
    if 'caps_width' in whis_kwargs.keys():
        caps_width = whis_kwargs.pop('caps_width')
    else:
        caps_width = .4
    if 'width' in mean_kwargs.keys():
        mean_width = mean_kwargs.pop('width')
    else:
        mean_width = 1.
    if 'width' in median_kwargs.keys():
        median_width = median_kwargs.pop('width')
    else:
        median_width = 1.
    if 'linestyle' not in mean_kwargs.keys():
        if median is not None:
            mean_kwargs['linestyle'] = '--'

    l1 = mpatches.Patch(color=color, alpha=alpha)
    l2 = ax.hlines([], [], [], **median_kwargs)
    l3 = ax.hlines([], [], [], **mean_kwargs)
    if return_handle:
        return (l1, l2, l3)

    x0, x1 = pos - .5*width, pos + .5*width
    if box is not None:  # plot box
        patch = PatchCollection(
            [Rectangle((x0, box[0]), width, box[1] - box[0])],
            facecolor=color,
            alpha=alpha,
            zorder=zorder)
        ax.add_collection(patch)

    if median is not None:
        x0_median = x0 + (1 - median_width)*width*.5
        x1_median = x1 - (1 - median_width)*width*.5
        ax.hlines(median, x0_median, x1_median, zorder=zorder, **median_kwargs)

    if mean is not None:  # plot mean
        x0_mean = x0 + (1 - mean_width)*width*.5
        x1_mean = x1 - (1 - mean_width)*width*.5
        ax.hlines(mean, x0_mean, x1_mean, zorder=zorder, **mean_kwargs)

    if whis is not None:  # plot whiskers
        if box is None:
            box = (whis[0], whis[0])
        ax.vlines((pos, pos), (whis[0], box[1]), (box[0], whis[1]),
                  zorder=zorder, **whis_kwargs)
        if showcaps:  # plot caps
            x0, x1 = pos - .5*caps_width*width, pos + .5*caps_width*width
            ax.hlines(whis, (x0, x0), (x1, x1), zorder=zorder, **whis_kwargs)

    return (l1, l2, l3)
