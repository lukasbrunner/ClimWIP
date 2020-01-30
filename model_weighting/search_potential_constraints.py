#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import argparse
import logging
import traceback
import numpy as np
import xarray as xr
from scipy import stats
# from sklearn.linear_model import TheilSenRegressor as TSR
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# from sklearn import linear_model
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  RidgeCV,
                                  Lasso,
                                  LassoCV,
                                  # RandomizedLasso,
                                  BayesianRidge,
                                  TheilSenRegressor)
from sklearn.feature_selection import f_regression

from core import utils
from core.get_filenames import get_filenames
from core.diagnostics import calculate_diagnostic
from core.read_config import read_config
from core.utils_xarray import area_weighted_mean


logger = logging.getLogger(__name__)


def read_args():
    """Read the given configuration from the config file"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        dest='config', nargs='?', default='DEFAULT',
        help='Name of the configuration to use (optional).')
    parser.add_argument(
        '--filename', '-f', dest='filename', default='configs/config.ini',
        help='Relative or absolute path/filename.ini of the config file.')
    return parser.parse_args()


def calc_target(filenames, cfg):
    # build and create path
    base_path = os.path.join(cfg.save_path, cfg.target_diagnostic)
    os.makedirs(base_path, exist_ok=True)

    targets = []
    for model_ensemble, filename in filenames[cfg.target_diagnostic].items():
        target = calculate_diagnostic(
            infile=filename,
            diagn=cfg.target_diagnostic,
            id_=model_ensemble.split('_')[2],
            base_path=base_path,
            time_period=(
                cfg.target_startyear,
                cfg.target_endyear),
            season=cfg.target_season,
            time_aggregation=cfg.target_agg,
            mask_land_sea=cfg.target_mask,
            region=cfg.target_region,
            overwrite=cfg.overwrite,
            idx_lats=None,
            idx_lons=None,
        )

        # calculate change rather than absolute value
        if cfg.target_startyear_ref is not None:
            target_hist = calculate_diagnostic(
                infile=filename,
                id_=model_ensemble.split('_')[2],
                diagn=cfg.target_diagnostic,
                base_path=base_path,
                time_period=(
                    cfg.target_startyear_ref,
                    cfg.target_endyear_ref),
                season=cfg.target_season,
                time_aggregation=cfg.target_agg,
                mask_land_sea=cfg.target_mask,
                region=cfg.target_region,
                overwrite=cfg.overwrite,
                idx_lats=None,
                idx_lons=None,
            )
            target[cfg.target_diagnostic] -= target_hist[cfg.target_diagnostic]

        target['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        targets.append(target)
        logger.debug('Calculate diagnostics for file {}... DONE'.format(filename))

    return area_weighted_mean(xr.concat(targets, dim='model_ensemble')[cfg.target_diagnostic])


def calc_predictor(filenames, idx, cfg):
    diagn = cfg.performance_diagnostics[idx]

    base_path = os.path.join(cfg.save_path, diagn)
    os.makedirs(base_path, exist_ok=True)

    varn = [*diagn.values()][0][0] if isinstance(diagn, dict) else diagn
    diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

    diagnostics = []
    for model_ensemble, filename in filenames[varn].items():
        diagnostic = calculate_diagnostic(
            infile=filename,
            id_=model_ensemble.split('_')[2],
            diagn=diagn,
            base_path=base_path,
            time_period=(
                cfg.performance_startyears[idx],
                cfg.performance_endyears[idx]),
            season=cfg.performance_seasons[idx],
            time_aggregation=cfg.performance_aggs[idx],
            mask_land_sea=cfg.performance_masks[idx],
            region=cfg.performance_regions[idx],
            overwrite=cfg.overwrite,
            idx_lats=None,
            idx_lons=None,
        )

        diagnostic['model_ensemble'] = xr.DataArray([model_ensemble], dims='model_ensemble')
        diagnostics.append(diagnostic)

    return area_weighted_mean(xr.concat(diagnostics, dim='model_ensemble')[diagn_key])


def calc_obs(diagn, idx, cfg):
    base_path = os.path.join(cfg.save_path, diagn)

    varn = [*diagn.values()][0][0] if isinstance(diagn, dict) else diagn
    diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

    obs_list = []
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
        filename = os.path.join(obs_path, f'{varn}_mon_{obs_id}_g025.nc')

        if not os.path.isfile(filename):
            continue

        try:
            obs = calculate_diagnostic(
                infile=filename,
                diagn=diagn,
                base_path=base_path,
                time_period=(
                    cfg.performance_startyears[idx],
                    cfg.performance_endyears[idx]),
                season=cfg.performance_seasons[idx],
                time_aggregation=cfg.performance_aggs[idx],
                mask_land_sea=cfg.performance_masks[idx],
                region=cfg.performance_regions[idx],
                overwrite=cfg.overwrite,
                idx_lats=None,
                idx_lons=None,
            )
            obs['obs_id'] = xr.DataArray([obs_id], dims='obs_id')
            obs_list.append(obs)
        except Exception:
            pass

    if len(obs_list) > 0:
        return area_weighted_mean(xr.concat(obs_list, dim='obs_id')[diagn_key]) if len(obs_list) > 0 else None


def plot(xx, yy, obs, reg, idx, cfg):
    colors_obs = {key: color for key, color in zip(
        ['ERA5', 'ERA-Interim', 'MERRA2'],
        list(sns.color_palette('colorblind', 3)))
    }

    units = {
        'clt': '%',
        'evspsbl': 'kg m-2 s-1',
        'hfss': 'W m-2',
        'hfls': 'W m-2',
        'hurs': '%',
        'huss': 'kg/kg',
        'pr': 'mm/day',
        'prw': 'kg m-2',
        'psl': 'Pa',
        'rlds': 'W m-2',
        'rldscs': 'W m-2',
        'rlus': 'W m-2',
        'rlut': 'W m-2',
        'rlutcs': 'W m-2',
        'rsds': 'W m-2',
        'rsdscs': 'W m-2',
        'rsdt': 'W m-2',
        'rsus': 'W m-2',
        'rsuscs': 'W m-2',
        'rsut': 'W m-2',
        'rsutcs': 'W m-2',
        'rtmt': 'W m-2',
        'sftlf': '%',
        'siconc': '%',
        'tauu': 'Pa',
        'tauv': 'Pa',
        'tas': 'degC',
        'tasmax': 'degC',
        'tasmin': 'degC',
        'tos': 'decC',
        'zg500': 'm',
    }

    fig, ax = plt.subplots()

    ax.scatter(xx, yy, color='k')
    ff = lambda x: reg.intercept + reg.slope*x
    ax.plot([min(xx), max(xx)], [ff(min(xx)), ff(max(xx))], color='k')
    if obs is not None:
        for id_ in obs['obs_id'].data:
            ax.axvline(obs.sel(obs_id=id_), color=colors_obs[id_], label=id_)
        ax.legend()

    unit = units[f'{cfg.performance_diagnostics[idx]}']
    if cfg.performance_aggs[idx] == 'TREND':
        unit += '/year'
    xlabel = ''.join([
        cfg.performance_diagnostics[idx],
        f' {cfg.performance_aggs[idx]}',
        f' {cfg.performance_startyears[idx]}-{cfg.performance_endyears[idx]}',
        f' ({unit})',
    ])
    ax.set_xlabel(xlabel)
    min_ = min(xx)
    max_ = max(xx)
    if obs is not None:
        min_ = min([min_, min(obs)])
        max_ = max([max_, max(obs)])
    offset = abs(max_ - min_) * .05
    ax.set_xlim(min_ - offset, max_ + offset)

    unit = units[f'{cfg.target_diagnostic}']
    if cfg.target_agg == 'TREND':
        unit += '/year'
    ylabel = ''.join([
        cfg.target_diagnostic,
        f' {cfg.target_agg}',
        f' {cfg.target_startyear}-{cfg.target_endyear}',
        f' minus {cfg.target_startyear_ref}-{cfg.target_endyear_ref}' if cfg.target_startyear_ref is not None else '',
        f' ({unit})'
    ])
    ax.set_ylabel(ylabel)
    offset = abs(max(yy) - min(yy)) * .05
    ax.set_ylim(min(yy) - offset, max(yy) + offset)

    sig = ''
    if reg.pvalue < .05:
        sig += '*'
    elif reg.pvalue < .01:
        sig += '*'
    ax.set_title(f'R$^2$={reg.rvalue**2:.2f}{sig}')

    fn = ''.join([
        'Correlation_',
        cfg.target_diagnostic, cfg.target_agg, '_',
        cfg.performance_diagnostics[idx], cfg.performance_aggs[idx],
        str(cfg.performance_startyears[idx]), '-', str(cfg.performance_endyears[idx]),
        '.png'
    ])

    os.makedirs(os.path.join(cfg.plot_path, cfg.config), exist_ok=True)
    plt.savefig(os.path.join(cfg.plot_path, cfg.config, fn), dpi=300)
    plt.close()


def calc_correlation(predictor, target, obs):

    def _theil_sen(xx, yy):
        """

        Returns
        -------
        coef_ : float
        """
        reg = TheilSenRegressor().fit(xx, yy)
        return reg.score(xx, yy)

    def _f_regression(xx, yy):
        """
        Univariate linear regression tests.

        Returns
        -------
        ff : float
            F values of features.
        pvalue : float
            p-values of F-scores.
        """
        return f_regression(xx, yy, center=True)[0][0]

    scores = {}
    reg = stats.linregress(predictor, target)
    scores['stats.linregress'] = reg.rvalue**2
    scores['sklearn.TheilSenRegressor'] = _theil_sen(predictor.reshape(-1, 1), target)
    scores['sklearn.f_regression'] = _f_regression(predictor.reshape(-1, 1), target)

    return scores, reg


def main():
    cfg = read_config(args.config, args.filename)

    diagns = cfg.performance_diagnostics

    scores_list = []
    nr_models = []
    pointless = []
    nr_obs = []
    cfg.independence_diagnostics = None
    for idx, diagn in enumerate(diagns):
        key = f'{diagn}{cfg.performance_aggs[idx]}'
        try:
            logger.info(f'Process {key}')
            # small hack to get the full set of filenames for target and
            # one diagnostic (as opposed to the subset which contains all
            # diagnostics).
            cfg.performance_diagnostics = [diagn]
            filenames, unique_models = get_filenames(cfg)
            cfg.performance_diagnostics = diagns

            target = calc_target(filenames, cfg)
            predictor = calc_predictor(filenames, idx, cfg)
            try:
                obs = calc_obs(diagn, idx, cfg)
            except Exception:
                obs = None

            # make sure order is the same
            target = target.sel(model_ensemble=predictor['model_ensemble'].data)
            assert tuple(target['model_ensemble'].data) == tuple(predictor['model_ensemble'].data)

            scores, reg = calc_correlation(predictor.data, target.data, obs)
            plot(predictor.data, target.data, obs, reg, idx, cfg)

            if obs is not None:
                if min(obs) > max(predictor) or max(obs) < min(predictor):
                    pointless.append(' x ')
                elif min(obs) < min(predictor) and max(obs) > max(predictor):
                    pointless.append(' x ')
                else:
                    pointless.append('   ')
                nr_obs.append(len(obs['obs_id']))
            else:
                pointless.append('   ')
                nr_obs.append(0)

            scores_list.append(scores)
            nr_models.append(len(filenames[cfg.target_diagnostic]))
        except Exception:
            cfg.performance_diagnostics = diagns
            scores_list.append(None)
            nr_models.append(0)
            pointless.append('   ')
            nr_obs.append(0)
            logger.error(f'Unexpected error encountered in {diagn}')
            logger.error(traceback.format_exc())

    r2_list = [xx['stats.linregress'] if xx is not None else 0 for xx in scores_list]
    ts_list = [xx['sklearn.TheilSenRegressor'] if xx is not None else 0 for xx in scores_list]
    fr_list = [xx['sklearn.f_regression'] if xx is not None else 0 for xx in scores_list]
    sort_idx = np.argsort(r2_list)[::-1]

    def get_rank_from_scores(scores_list):
        r2 = [xx['stats.linregress'] if xx is not None else 0 for xx in scores_list]
        ts = [xx['sklearn.TheilSenRegressor'] if xx is not None else 0 for xx in scores_list]
        fr = [xx['sklearn.f_regression'] if xx is not None else 0 for xx in scores_list]

        points1 = np.argsort(np.argsort(r2))
        points2 = np.argsort(np.argsort(ts))
        points3 = np.argsort(np.argsort(fr))

        points = [p1 + p2 + p3 for (p1, p2, p3) in zip(points1, points2, points3)]

        return np.argsort(points)[::-1]

    sort_idx = get_rank_from_scores(scores_list)

    lines = OrderedDict()
    rank = 1
    for idx in sort_idx:
        if r2_list[idx] > .01:
            key = f'{cfg.performance_diagnostics[idx]}{cfg.performance_aggs[idx]}'
            lines[key] = ' | '.join([
                f'{rank:<4}',
                f'{key:<12}',
                f'{nr_models[idx]:<7}',
                f'{r2_list[idx]:.3f} ',
                f'{ts_list[idx]:+.3f}  ',
                f'{fr_list[idx]:<6.3f}',
                f' {nr_obs[idx]} ',
                f'{pointless[idx]}',
                ''
            ])
            rank += 1

    for idx1, diagn1 in enumerate(diagns):
        key1 = f'{diagn1}{cfg.performance_aggs[idx1]}'

        if key1 not in lines.keys():
            continue

        for idx2, diagn2 in enumerate(diagns):
            key2 = f'{diagn2}{cfg.performance_aggs[idx2]}'

            if key1 == key2:
                continue

            try:
                logger.info(f'Process {key1} and {key2}')

                # now I want target and two diagnostics
                cfg.performance_diagnostics = [diagn1, diagn2]
                filenames, unique_models = get_filenames(cfg)
                cfg.performance_diagnostics = diagns

                predictor1 = calc_predictor(filenames, idx1, cfg)
                predictor2 = calc_predictor(filenames, idx2, cfg)

                predictor2 = predictor2.sel(model_ensemble=predictor1['model_ensemble'].data)
                assert tuple(predictor1['model_ensemble'].data) == tuple(predictor2['model_ensemble'].data)

                _, reg = calc_correlation(predictor1.data, predictor2.data, None)

                if reg.rvalue**2 > .85:
                    lines[key1] += f'{key2}, '

            except Exception:
                pass
                # cfg.performance_diagnostics = diagns
                # logger.error(traceback.format_exc())

    print('rank | varAGG       | #models | linear | TheilSen | F-Reg. | obs | bad | correlated to ')
    print('\n'.join([line for line in lines.values()]))
    with open(os.path.join(cfg.save_path, f'{cfg.config}_possible_constraints.txt'), 'w') as ff:
        ff.write('rank | varAGG       | #models | linear | TheilSen | F-Reg. | obs | bad | correlated to\n')
        ff.write('\n'.join([line for line in lines.values()]))


if __name__ == '__main__':
    args = read_args()
    utils.set_logger()
    main()
