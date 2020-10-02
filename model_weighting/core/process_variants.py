#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import logging
import numpy as np
import xarray as xr
from copy import copy
from natsort import natsorted

logger = logging.getLogger(__name__)


def get_model_variants(model_ensemble):
    """
    Get a nested list of model variants.

    Parameters
    ----------
    model_ensemble : array-like, shape (N,)

    Returns
    -------
    model_ensemble_nested : list of lists, shape (M,)
        Each sub-list contains all variants of the same model ordered by
        natsort. If there is only one variant per model M=N otherwise M<N.
    """
    models = np.array(['_'.join([me.split('_')[0], me.split('_')[2]])
                       for me in model_ensemble])
    model_ensemble_nested = []
    for model in natsorted(np.unique(models)):
        idx = np.where(models == model)[0]
        model_ensemble_nested.append(natsorted(np.array(model_ensemble)[idx]))
    return model_ensemble_nested


def _tile(data, reps=None):
    """Tile means to fill into each variant."""
    return np.tile(data, (reps, 1))


def _set_diagonal(data):
    """Set the diagonal elements to nan again"""
    data = copy(data)  # the original array is write-protected
    np.fill_diagonal(data, np.nan)  # NOTE: this works in place!
    return data


# NOTE: this function is not used in the current implementation
def spread_to_weight(ratios, metric='logistic', **kwargs):
    """
    Implements different metrics to calculate weights based on the spread ratios.

    Parameters
    ----------
    ratios : array-like, shape (N,)
    metric : string {'logistic', 'linear'}, optional
        - 'linear': Weights are zero below threshold_0, one above threshold_1
          and linearly increasing in between.
        - 'logistic': Weights follow a logistic function with given parameters
    **kwargs : dict, optional
        Valid keyword arguments to be passed on the function defined by metric.

    Returns
    -------
    weights : numpy.array, shape (N,)
    """
    def _equal(ratios):
        """
        Disregards information from spread and weights each diagnostic equally.
        """
        return np.ones_like(ratios)

    def _linear(ratios, best=.1, worst=.5):
        """
        A weighting function with removable discontinuities at best and worst.
        """
        ff = lambda x: 1 - ((x - best) / (worst - best))
        weights = ff(ratios)
        weights[weights > 1] = 1  # set to best weight
        weights[weights < 0] = 0  # set to zero
        return weights

    def _logistic(xx, B=-7, Q=2):
        """
        A generalised logistic function tailored to the application here.

        Maps: f(x): [0, 1] -> (0, 1) with the special properties:
        - smaller values get higher weights
        - Asymptotic at both end but not symmetric

        Parameters
        ----------
        xx : array-like, shape (N,)
            Array of input values
        B : float < 0, optional
            Growth rate.
        Q : float > 0, optional
            Symmetry parameter.

        Fixed parameters
        --------------------
        (move to function call if you want to use them)
        A : float
            Lower asymptote.
        K : float
            Upper asymptote.
        nu : float
            Not quite clear what that does...
        M : float
            Center value. Can be interpreted as starting time.

        Returns
        -------
        yy : numpy.array, shape (N,)

        Links
        -----
        https://en.wikipedia.org/wiki/Generalised_logistic_function
        """
        if np.any(xx < 0.):
            errmsg = 'xx must be positive'
            logger.error(errmsg)
            raise ValueError(errmsg)
        if np.any(xx > 1.):
            idx = np.where(xx > 1.)
            logmsg = ' '.join([
                f'Spread ratio > 1 for diagnostics number: {idx}! These',
                'should probably really not be used!'])
            logger.warning(logmsg)
        if B > 0.:
            errmsg = ' '.join([
                'B > 0: positive groth rates lead to higher weights for worse',
                'diagnostics! If you really want this comment this test.'])
            logger.error(errmsg)
            raise ValueError(errmsg)
        if Q < 0.:
            errmsg = 'Negative symmetry parameters are not allowed!'
            logger.error(errmsg)
            raise ValueError(errmsg)
        xx = np.array(xx)
        A = .1  # lower asymptote
        K = 1  # upper asymptote
        u = .5  # ?
        M = .5  # center (starting time)
        return A + ((K-A) / ((1 + Q*np.exp(-B*(xx-M))**(1/u))))

    # DEBUG: plot the function
    # import matplotlib.pyplot as plt
    # xx = np.linspace(0, 1, 100)
    # plt.plot(_logistic(xx))
    # plt.show()
    # ------

    if metric == 'equal':
        return _equal(ratios)
    elif metric == 'linear':
        return _linear(ratios, **kwargs)
    elif metric == 'logistic':
        return _logistic(ratios, **kwargs)
    else:
        raise ValueError


def _mean(data, weights):
    """Weighted average of diagnostics."""
    return np.average(data, weights=weights)


def independence_sigma_from_variants(delta_i, delta_i_temp, model_ensemble_nested):
    """
    NOTE: all of this is a bit add hoc! The idea it the following
    - if a model-model distance is similar to the distance between variants
      (~variant_std) the corresponding value from the gaussian should be about 1
      (i.e., the model should be recognized as copy)
    - if a model-model distance is much larger than the mean model-model distance
      (~model_mean + model_std) the corresponding value should be about 0
      (i.e., the model should be recognized as very independent)
    """
    # the mean distance between models
    delta_i_models = delta_i.mean('model_ensemble', skipna=True).mean('perfect_model_ensemble')

    delta_i_variants = []
    for model_ensemble in model_ensemble_nested:
        if len(model_ensemble) == 1:
            continue

        model_id = '_'.join([
            model_ensemble[0].split('_')[0],
            str(len(model_ensemble)),
            model_ensemble[0].split('_')[2]])

        # the mean distance of variants of the same model
        delta_i_variants.append(delta_i_temp.sel(
            model_ensemble=model_id,
            perfect_model_ensemble=model_ensemble[1:]
        ).mean('perfect_model_ensemble').data)

    # mean over all models with more than one variant
    delta_i_variants = np.mean(delta_i_variants)

    # sigma_i should be > delta_i_variants to that models which are
    # in the spread of the variants are recognized as potential copies
    # but not so large that all models are dependent and the weighting
    # becomes pointless
    # NOTE: a bit add hoc: I thake the mean between the two at the moment
    sigma_i = np.atleast_1d(np.mean([delta_i_models, delta_i_variants]))

    return sigma_i


def process_variants(da, cfg):
    """
    Handle operations which need to be aware of model variants.

    The functions here are aware of the difference between different models
    and variants of the same model as identified by the sub-routine
    get_model_variants.

    Parameters
    ----------
    da : xarray.DataArray, shape (N,M) or (N,N,M)
        An array containing the model-observation (N,) or model-model (N,N)
       distances for all diagnostics (M,).
    cfg : config object

    Returns
    -------
    da : xarray.DataArray, shape same as input
        An array where all values of variants of the same model are replaced
        by their respective mean value.
    diagnostic_weights : xarray.DataArray, shape (M,)
        An array containing a weight for each diagnostic based on its quality
        as estimated from the model variants. NOTE: if all models have only one
        variant the weights will all be set to 1.
    """
    if 'perfect_model_ensemble' in da.dims:
        # model-model matrix
        inter_model = True
    else:
        # model-observation vector
        inter_model = False

    # user given weights per diagnostic
    if inter_model and cfg.independence_weights is not None:
        diagnostic_weights_user = np.array(cfg.independence_weights)
    elif not inter_model and cfg.performance_weights is not None:
        diagnostic_weights_user = np.array(cfg.performance_weights)
    else:  # no user weights
        diagnostic_weights_user = np.ones_like(da['diagnostic'])

    # make sure they are normalized
    diagnostic_weights_user = diagnostic_weights_user / diagnostic_weights_user.sum()

    model_ensemble_nested = get_model_variants(da['model_ensemble'].data)

    variants_std = []
    da_list = []
    for model_ensemble in model_ensemble_nested:
        if len(model_ensemble) == 1:
            da_list.append(da.sel(model_ensemble=model_ensemble))
            continue  # skip models with only one variant

        # select all variants
        da_sel = da.sel(model_ensemble=model_ensemble)

        # get mean and standard deviation over all variants of the same model
        mean_ = da_sel.mean('model_ensemble', skipna=True)
        std_ = da_sel.std('model_ensemble', skipna=True)

        if inter_model:
            # If there are only two model variants they have only one non-nan value
            # in the perfect model setting since the other one is from the model
            # to itself and had been set to nan in the distance matrix!
            # In this case we can not use the standard deviation
            std_.data[std_.data == 0.] = np.nan

        model_id = '_'.join([
            model_ensemble[0].split('_')[0],
            str(len(model_ensemble)),
            model_ensemble[0].split('_')[2]])
        da_list.append(mean_.expand_dims({'model_ensemble': [model_id]}))
        variants_std.append(std_.expand_dims({'model_ensemble': [model_id]}))

    da_mean = xr.concat(da_list, dim='model_ensemble')
    variant_std = xr.concat(variants_std, dim='model_ensemble').mean('model_ensemble')
    model_std = da_mean.std('model_ensemble')

    if inter_model:  # do the same for the perfect model dimension
        # this still contains all variant in the perfect model dimension for the sigma_i calculation
        da_mean_temp = da_mean.copy()
        da_list = []
        variants_std = []
        for model_ensemble in model_ensemble_nested:
            if len(model_ensemble) == 1:
                da_list.append(da_mean.sel(perfect_model_ensemble=model_ensemble))
                continue  # skip models with only one variant
            da_sel = da_mean.sel(perfect_model_ensemble=model_ensemble)
            da_std_sel = variant_std.sel(perfect_model_ensemble=model_ensemble)
            mean_ = da_sel.mean('perfect_model_ensemble', skipna=True)
            std_ = da_std_sel.mean('perfect_model_ensemble', skipna=True)
            model_id = '_'.join([
                model_ensemble[0].split('_')[0],
                str(len(model_ensemble)),
                model_ensemble[0].split('_')[2]])
            da_list.append(mean_.expand_dims({'perfect_model_ensemble': [model_id]}))
            variants_std.append(std_.expand_dims({'perfect_model_ensemble': [model_id]}))

        da_mean = xr.concat(da_list, dim='perfect_model_ensemble')
        variant_std = xr.concat(variants_std, dim='perfect_model_ensemble').mean('perfect_model_ensemble')
        model_std = da_mean.std('model_ensemble').mean('perfect_model_ensemble')

    # NOTE: this is not used in the current implementation
    # the spread ratio is an estimate of the quality of a predictor; the larger it is
    # the higher is the effect of internal variability in a predictor - we therefore
    # prefer predictors with a low spread ratio
    spread_ratios = variant_std / model_std
    diagnostic_weights_quality = xr.apply_ufunc(
        spread_to_weight, spread_ratios,
        kwargs={'metric': 'equal'})  # TODO: if we want to change this we need an additional flag

    # combine them with user given weights
    diagnostic_weights = diagnostic_weights_quality * diagnostic_weights_user
    diagnostic_weights = diagnostic_weights / diagnostic_weights.sum('diagnostic')

    if len(da_mean['diagnostic']) > 1:
        diagnostic = xr.apply_ufunc(
            _mean, da_mean, diagnostic_weights,
            input_core_dims=[['diagnostic'], ['diagnostic']],
            vectorize=True)
    else:
        diagnostic = da_mean.squeeze()

    model_ensemble = natsorted(diagnostic['model_ensemble'].data)
    diagnostic = diagnostic.sel(model_ensemble=model_ensemble)
    da_mean = da_mean.sel(model_ensemble=model_ensemble)
    if inter_model:
        # make sure they are sorted the same way
        diagnostic = diagnostic.sel(perfect_model_ensemble=model_ensemble)
        da_mean = da_mean.sel(perfect_model_ensemble=model_ensemble)
        # set the diagonal elements to nan again (have been overwritten by mean)
        diagnostic = xr.apply_ufunc(
            _set_diagonal, diagnostic,
            input_core_dims=[['model_ensemble', 'perfect_model_ensemble']],
            output_core_dims=[['model_ensemble', 'perfect_model_ensemble']],
            vectorize=True)

        da_mean = xr.apply_ufunc(
            _set_diagonal, da_mean,
            input_core_dims=[['model_ensemble', 'perfect_model_ensemble']],
            output_core_dims=[['model_ensemble', 'perfect_model_ensemble']],
            vectorize=True)

        if cfg.variants_independence:
            if len(da_mean['diagnostic']) > 1:
                diagnostic_temp = xr.apply_ufunc(
                    _mean, da_mean_temp, diagnostic_weights,
                    input_core_dims=[['diagnostic'], ['diagnostic']],
                    vectorize=True)
            else:
                diagnostic_temp = da_mean_temp.squeeze()
            sigma_i = independence_sigma_from_variants(diagnostic, diagnostic_temp, model_ensemble_nested)
        else:
            sigma_i = None
    else:
        sigma_i = None

    for idx, _ in enumerate(diagnostic_weights):
        if inter_model:
            diagn = ' '.join([
                'independence diagnostic',
                f'{cfg.independence_diagnostics[idx]}{cfg.independence_aggs[idx]}'])
        else:
            diagn = ' '.join([
                'performance diagnostic',
                f'{cfg.performance_diagnostics[idx]}{cfg.performance_aggs[idx]}'])

        logmsg = ' '.join([
            f'Spread ratio -> quality weight by "equal"',
            f'x user weight -> total weight for {diagn}:',
            f'{spread_ratios.data[idx]:.2f} -> {diagnostic_weights_quality.data[idx]:.2f} x',
            f'{diagnostic_weights_user.data[idx]:.2f} -> {diagnostic_weights.data[idx]:.2f}'])
        if spread_ratios[idx] < .7:
            logger.info(logmsg)
        else:
            logger.warning(f'{logmsg} Consider not using this diagnostic?')

    if (np.all([len(me) == 1 for me in model_ensemble_nested]) or
        not cfg.variants_combine):
        if len(da['diagnostic']) > 1:
            delta_i = xr.apply_ufunc(
                _mean, da, diagnostic_weights_user,
                input_core_dims=[['diagnostic'], ['diagnostic']],
                vectorize=True)
        else:
            delta_i = da.squeeze()
        delta_i.name = 'delta_i'
        return delta_i, sigma_i, da

    return diagnostic, sigma_i, da_mean


def process_variants_target(da, cfg):
    """
    Handle operations which need to be aware of model variants.

    A simpler version of process_variants only for 1D arrays.

    Parameters
    ----------
    da : xarray.DataArray, shape (N,)
        An array containing the target values.
    cfg : config object

    Returns
    -------
    da : xarray.DataArray, shape (M<=N,)
        An array where all models with more than one variant are reduced
        to the respective mean value.
    """
    model_ensemble_nested = get_model_variants(da['model_ensemble'].data)

    if np.all([len(me) == 1 for me in model_ensemble_nested]) or not cfg.variants_combine:
        return da

    da_list = []
    for model_ensemble in model_ensemble_nested:
        if len(model_ensemble) == 1:
            da_list.append(da.sel(model_ensemble=model_ensemble))
            continue  # skip models with only one variant

        da_sel = da.sel(model_ensemble=model_ensemble)
        mean_ = da_sel.mean('model_ensemble', skipna=True)
        model_id = '_'.join([
            model_ensemble[0].split('_')[0],
            str(len(model_ensemble)),
            model_ensemble[0].split('_')[2]])
        da_list.append(mean_.expand_dims({'model_ensemble': [model_id]}))

    da_mean = xr.concat(da_list, dim='model_ensemble')
    da_mean = da_mean.sel(model_ensemble=natsorted(da_mean['model_ensemble'].data))
    return da_mean


def expand_variants(ds, model_ensemble):
    model_ensemble_nested = get_model_variants(model_ensemble)
    counts = [len(me) for me in model_ensemble_nested]
    ds['variant_count'] = xr.DataArray(counts, dims='model_ensemble')

    def _expand_variants_data(data, model_ensemble_org):
        data_expanded = []
        for idx, model_ensemble in enumerate(model_ensemble_nested):
            model1 = model_ensemble_org[idx].split('_')[0]
            model2 = model_ensemble[0].split('_')[0]
            assert model1 == model2
            data_expanded += [data[idx]] * len(model_ensemble)
        return np.array(data_expanded)

    da_list = []
    for varn in ds:
        if 'model_ensemble' not in ds[varn].dims:
            continue

        da = xr.apply_ufunc(
            _expand_variants_data, ds[varn], ds['model_ensemble'],
            input_core_dims=[['model_ensemble'], ['model_ensemble']],
            output_core_dims=[['temp']],
            vectorize=True).rename({'temp': 'model_ensemble'})
        da['model_ensemble'] = xr.DataArray(model_ensemble, dims='model_ensemble')

        if 'perfect_model_ensemble' in ds[varn].dims:
            da = xr.apply_ufunc(
                _expand_variants_data, da, da['perfect_model_ensemble'],
                input_core_dims=[['perfect_model_ensemble'], ['perfect_model_ensemble']],
                output_core_dims=[['temp']],
                vectorize=True).rename({'temp': 'perfect_model_ensemble'})
            da['perfect_model_ensemble'] = xr.DataArray(model_ensemble, dims='perfect_model_ensemble')
        da.name = varn
        da_list.append(da)
        ds = ds.rename({varn: f'{varn}_mean'})

    ds_expanded = xr.merge(da_list)
    ds_expanded['weights'].data /= ds_expanded['variant_count'].data
    ds_expanded['weights'].data /= ds_expanded['weights'].sum('model_ensemble').data
    ds = ds.rename({'model_ensemble': 'model', 'perfect_model_ensemble': 'perfect_model'})
    return xr.merge([ds, ds_expanded])
