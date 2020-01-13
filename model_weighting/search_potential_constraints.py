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
from sklearn.linear_model import TheilSenRegressor as TSR
import matplotlib.pyplot as plt
import seaborn as sns

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

    varn = [*diagn.values()][0, 0] if isinstance(diagn, dict) else diagn
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

    varn = [*diagn.values()][0, 0] if isinstance(diagn, dict) else diagn
    diagn_key = [*diagn.keys()][0] if isinstance(diagn, dict) else diagn

    obs_list = []
    for obs_path, obs_id in zip(cfg.obs_path, cfg.obs_id):
        filename = os.path.join(obs_path, f'{varn}_mon_{obs_id}_g025.nc')

        if not os.path.isfile(filename):
            continue

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
    offset = abs(max(xx) - min(xx)) * .05
    ax.set_xlim(min(xx) - offset, max(xx) + offset)

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

    plt.savefig(os.path.join(cfg.plot_path, fn), dpi=300)
    plt.close()


def calc_correlation(predictor, target, obs):

    def _ols(xx, yy):
        return stats.linregress(xx, yy)

    def _theil_sen(xx, yy):
        return stats.theilslopes(yy, xx)

    def _theil_sen2(xx, yy):
        reg = TSR().fit(xx, yy)
        import ipdb; ipdb.set_trace()
        return reg.interc_, reg.coef_[0], reg.score(xx, yy)

    reg = stats.linregress(predictor, target)
    # tsr1 = _theil_sen(predictor.data, target.data)
    # tsr2 = _theil_sen2(predictor.data.reshape(-1, 1), target.data)

    return reg


def main():
    cfg = read_config(args.config, args.filename)

    diagns = cfg.performance_diagnostics

    r2 = []
    pp = []
    for idx, diagn in enumerate(diagns):
        try:
            logger.info(f'Process {diagn} {cfg.performance_aggs[idx]}')
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

            reg = calc_correlation(predictor.data, target.data, obs)
            plot(predictor.data, target.data, obs, reg, idx, cfg)

            r2.append(reg.rvalue**2)
            pp.append(reg.pvalue)
        except Exception:
            cfg.performance_diagnostics = diagns
            r2.append(0)
            pp.append(9)
            logger.error(f'Unexpected error encountered in {diagn}')
            logger.error(traceback.format_exc())

    sort_idx = np.argsort(r2)[::-1]

    line = ''
    line += 'Variable | Agg        | Period    | R2    | p\n'
    line += '-'*50 + '\n'
    for idx in sort_idx:
        line += ' | '.join([
            f'{cfg.performance_diagnostics[idx]:<8}',
            f'{cfg.performance_aggs[idx]:<11}',
            f'{cfg.performance_startyears[idx]}-{cfg.performance_endyears[idx]}',
            f'{r2[idx]:.3f}',
            f'{pp[idx]:.3f}\n',
        ])

    print(line)
    with open(os.path.join(cfg.save_path, f'{cfg.config}_possible_constraints.txt'), 'w') as ff:
        ff.write(line)

    # now we test correlation between predictors and make a list of highly correlated predictors for each predictor
    # (this could be done whay more elegant but I can't be bothered right now...
    line = ''
    for idx1, diagn1, in enumerate(diagns):
        for idx2, diagn2 in enumerate(diagns):
            if diagn1 == diagn2:
                continue
            try:
                logger.info(f'Process {diagn1} {cfg.performance_aggs[idx1]} and {diagn2} {cfg.performance_aggs[idx2]}')
                # now I want target and two diagnostics
                cfg.performance_diagnostics = [diagn1, diagn2]
                filenames, unique_models = get_filenames(cfg)
                cfg.performance_diagnostics = diagns

                predictor1 = calc_predictor(filenames, idx1, cfg)
                predictor2 = calc_predictor(filenames, idx2, cfg)

                predictor2 = predictor2.sel(model_ensemble=predictor1['model_ensemble'].data)
                assert tuple(predictor1['model_ensemble'].data) == tuple(predictor2['model_ensemble'].data)

                reg = calc_correlation(predictor1.data, predictor2.data, None)

                if reg.rvalue**2 > .85:
                    line += f'{diagn1:<10} {cfg.performance_aggs[idx1]:<5} | {diagn2:<8} {cfg.performance_aggs[idx2]:<5} | {reg.rvalue**2:.3f}\n'
                else:
                    pass
                    # line += f'{diagn1:<10} {cfg.performance_aggs[idx1]:<5} | {diagn2:<8} {cfg.performance_aggs[idx2]:<5} | {reg.rvalue**2:.3f}\n'

            except Exception:
                line += f'{diagn1:<10} {cfg.performance_aggs[idx1]:<5} | {diagn2:<8} {cfg.performance_aggs[idx2]:<5} | failed\n'
                cfg.performance_diagnostics = diagns
                logger.error(traceback.format_exc())

    print(line)
    with open(os.path.join(cfg.save_path, f'{cfg.config}_predictor_correlations.txt'), 'w') as ff:
        ff.write(line)


if __name__ == '__main__':
    args = read_args()
    utils.set_logger()
    main()
