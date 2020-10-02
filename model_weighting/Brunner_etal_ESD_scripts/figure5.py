#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2020-09-12 17:46:56 lukas>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial as spatial
from glob import glob
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from utils_python.xarray import area_weighted_mean
from utils_python.utils import get_first_member
from natsort import natsorted, ns


mpl.rcParams['lines.linewidth'] = 0.5

PATH = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/'
PLOTPATH = 'figures/'
SAVEPATH2 = '/home/lukbrunn/Documents/Projects/Paper_CMIP6/figures'

def extract_models_from_filenames(filenames):
    """
    Return the model name based on the given filenames.

    Parameters
    ----------
    filenames : str or list of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    models : str or ndarray of str based on input
    """
    if isinstance(filenames, str):
        return os.path.basename(filenames).split('_')[2]
    return np.array([os.path.basename(fn).split('_')[2] for fn in filenames])


def extract_variants_from_filenames(filenames):
    """
    Return the variant ID based on the given filenames.

    Parameters
    ----------
    filenames : str or list of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    variants : str or ndarray of str based on input
    """
    if isinstance(filenames, str):
        return os.path.basename(filenames).split('_')[4]
    return np.array([os.path.basename(fn).split('_')[4] for fn in filenames])


def cluster_by_models(filenames, return_idx=False, return_keys=True, return_filenames=False):
    """
    Return a nested list of indices separating variants of the same model.

    Parameters
    ----------
    filenames : str or list of str or dict of str
        A list of valid CMIP6ng filenames.

    Returns
    -------
    nested : list of lists
    """
    if return_keys and not isinstance(filenames, dict):
        raise ValueError('filenames has to be dict if return_keys is True')

    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(filenames, dict):
        model_variants = np.array([*filenames.keys()])
        models = np.array([model_variant.split('_')[0]
                           for model_variant in model_variants])
        filenames = [filenames[key] for key in model_variants]
    else:
        models = extract_models_from_filenames(filenames)

    idx_nested = []
    for model in natsorted(np.unique(models), alg=ns.IC):
        idx = np.where(models == model)[0]
        idx_nested.append(idx)
    if return_idx:
        return idx_nested
    elif return_filenames:
        return [[filenames[idx] for idx in idxs] for idxs in idx_nested]
    return [[model_variants[idx] for idx in idxs] for idxs in idx_nested]


def intersect_models(*args):
    """
    Return intersection of filenames available for all cases.

    Parameters
    ----------
    *args : One or more list of filenames

    Returns
    -------
    *args : tuple of dict, same length as input
        One ore more dictionaries of filenames available for all models
        (and variants) from the input lists. The dictionary keys are
        <model>_<variant>
    """
    filenames_dict = ()
    for filenames in args:
        models = extract_models_from_filenames(filenames)
        variants = extract_variants_from_filenames(filenames)
        filenames_dict += (
            {f'{model}_{variant}': filename
             for model, variant, filename in zip(models, variants, filenames)},)

    if len(args) == 1:
        return filenames_dict[0]

    for filenames in filenames_dict:
        try:
            intersected_model_variants = list(
                np.intersect1d(intersected_model_variants, list(filenames.keys())))
        except NameError:
            intersected_model_variants = list(filenames.keys())

    args = ()
    for filenames in filenames_dict:
        args += ({key: filenames[key] for key in intersected_model_variants},)

    return args


def get_model_ensemble(ds):
    if 'height' in ds:
        del ds['height']
    fn = ds.encoding['source']
    model_ensemble = '_'.join([fn.split('_')[ii] for ii in [2, 4]])
    ds['model_ensemble'] = xr.DataArray(model_ensemble, name='model_ensemble')
    return ds


def read_cmip5(varn='tas', period='1951-2000', region='GLOBAL', season='ANN', aggregation='CLIM',
               masked=False, ensembles=False):
    filename_pattern = f'{varn}/{varn}_mon_*_rcp85_*_g025_{period}_{season}_{aggregation}_{region}_unmasked.nc'
    if ensembles:
        filenames = os.path.join(PATH, filename_pattern)
    else:
        filenames = get_first_member(os.path.join(PATH, filename_pattern))

    filenames = glob(filenames)
    filenames = intersect_models(filenames)
    models = cluster_by_models(filenames)

    ds_list = []
    for model in models:
        ds_model = []
        for mm in model:
            ds = xr.open_dataset(filenames[mm])
            ds_model.append(ds)
        ds = xr.concat(ds_model, dim='ensemble').mean('ensemble')
        ds = ds.expand_dims({'model_ensemble': ['*' + mm.split('_')[0]]})
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='model_ensemble')

    return ds


def read_cmip6(varn='tas', period='1951-2000', region='GLOBAL', season='ANN', aggregation='CLIM',
               masked=False, ensembles=False):
    # filename_pattern = f'{varn}/{varn}_mon_*_ssp585_*_g025_{period}_{season}_{aggregation}_{region}_{"mask" if masked else "unmasked"}.nc'
    filename_pattern = f'{varn}/{varn}_mon_*_ssp585_*_g025_{period}_{season}_{aggregation}_{region}_unmasked.nc'
    if ensembles:
        filenames = os.path.join(PATH, filename_pattern)
    else:
        filenames = get_first_member(os.path.join(PATH, filename_pattern))

    filenames = glob(filenames)
    filenames = intersect_models(filenames)
    models = cluster_by_models(filenames)

    ds_list = []
    for model in models:
        if model[0].split('_')[0] not in [
                'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5-CanOE', 'CanESM5', 'CESM2-WACCM', 'CESM2', 'CNRM-CM6-1-HR', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MCM-UA-1-0', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-MM', 'UKESM1-0-LL']:
            continue
        ds_model = []
        for mm in model:
            ds = xr.open_dataset(filenames[mm])
            ds_model.append(ds)
        ds = xr.concat(ds_model, dim='ensemble').mean('ensemble')
        ds = ds.expand_dims({'model_ensemble': [mm.split('_')[0]]})
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='model_ensemble')

    return ds


def get_labels_colors(labels):
    """Color models due to MAsson and Knutti (2011)."""
    colors = sns.hls_palette(8, l=.4)
    label_colors = {}
    for label in labels:
        if ('ACCESS' in label.upper() or
            'HadGEM' in label or
            'KACE' in label.upper() or
            'UKESM' in label.upper()):
            label_colors[label] = colors[0]
        # elif 'GISS' in label.upper():
        #     label_colors[label] = colors[1]
        elif ('MPI' in label.upper() or
              # 'CAMS' in label.upper() or
              'NESM3' in label.upper() or
              'AWI' in label.upper()):
            label_colors[label] = colors[1]
        elif 'MIROC' in label.upper():
            label_colors[label] = colors[2]
        # elif 'IPSL' in label.upper():
        #     label_colors[label] = colors[4]
        # elif 'MRI' in label.upper():
        #     label_colors[label] = colors[5]
        elif ('CESM' in label.upper() or
              # 'bcc' in label.lower() or
              'NorESM' in label or
              'FIO' in label.upper()
              # or 'MCM' in label.upper()
        ):
            label_colors[label] = colors[3]
        # elif 'GFDL' in label.upper():
        #     label_colors[label] = colors[7]
        elif ('CNRM' in label.upper() or
              'EC-EARTH' in label.upper()):
            label_colors[label] = colors[4]
        elif ('CanESM' in label):
            label_colors[label] = colors[5]
        elif 'FGOALS' in label.upper():
            label_colors[label] = colors[6]
        elif ('INM-CM' in label):
            label_colors[label] = colors[7]
        else:
            label_colors[label] = 'k'
    return label_colors


def get_label_weights(labels):
    label_weights = {}
    for label in labels:
        if 'CMIP6' in label:
            label_weights[label] = 'bold'
        elif 'OBS' in label:
            label_weights[label] = 'bold'
        else:
            label_weights[label] = 'regular'
    return label_weights


def get_label_styles(labels):
    label_styles = {}
    for label in labels:
        if 'CMIP3' in label:
            label_styles[label] = 'italic'
        elif 'OBS' in label:
            label_styles[label] = 'italic'
        else:
            label_styles[label] = 'normal'
    return label_styles


def weighted_distance_matrix(data, lat=None):
    """An area-weighted RMS condensed distance matrix"""
    if lat is None:
        w_lat = np.ones(data.shape[-2])
    else:
        w_lat = np.cos(np.radians(lat))
    weights = np.tile(w_lat, (data.shape[-1], 1)).swapaxes(0, 1).ravel()
    weights /= weights.sum()
    data = data.reshape((data.shape[0], weights.shape[0]))

    # remove nans by weighting them with 0
    idxs = np.where(np.isnan(data))
    data_new = data.copy()
    data_new[idxs] = 0
    data = data_new
    weights[idxs[0]] = 0

    return pdist(data, metric='euclidean', w=weights)


def distance_matrix_(data):
    return pdist(data.reshape(-1, 1), metric='euclidean')


def main():
    varns = ['pr']  # ['tas', 'psl']
    period = '1980-2014'
    season = 'ANN'
    region = 'GLOBAL'
    aggregations = ['ANOM-GLOBAL']  # ['CLIM', 'CLIM']
    ensembles = True

    # varns = ['tas']
    # period = '1980-2014'
    # season = 'ANN'
    # region = 'GLOBAL'
    # aggregations = ['TREND']
    # ensembles = True

    if not isinstance(varns, list):
        varns = [varns]
    if not isinstance(aggregations, list):
        aggregations = [aggregations]

    ds_list = []
    varns_new = []
    model_ensemble = None
    for varn, aggregation in zip(varns, aggregations):
        ds = read_cmip6(varn=varn, period=period, season=season, aggregation=aggregation,
                        region=region, ensembles=ensembles)

        # ds5 = read_cmip5(varn=varn, period=period, season=season, aggregation=aggregation,
        #                 region=region, ensembles=ensembles)
        # ds = xr.merge([ds, ds5])

        varn_new = f'{varn}{aggregation}'
        ds = ds.rename({varn: varn_new})
        varns_new.append(varn_new)
        ds_list.append(ds)

        # need to check if ensemble is available for all variables
        if model_ensemble is None:
            model_ensemble = ds['model_ensemble'].data
        else:
            model_ensemble = np.intersect1d(model_ensemble, ds['model_ensemble'].data)

    ds = xr.merge(ds_list).load()

    varns = varns_new

    labels = ds['model_ensemble'].data
    labels = [label.split('_')[0] for label in labels]
    label_colors = get_labels_colors(labels)
    label_weights = get_label_weights(labels)
    label_styles = get_label_styles(labels)

    mips = []
    if np.any(['CMIP3' in label for label in labels]):
        mips.append('3')
    if np.any(['CMIP5' in label for label in labels]):
        mips.append('5')
    if np.any(['CMIP6' in label for label in labels]):
        mips.append('6')
    mip_str = 'CMIP' + '-'.join(mips)

    ens_str = '_ensembles' if ensembles else ''

    rmse_list = []
    for varn in varns:
        if aggregations[0] == 'TREND':
            rmse = xr.apply_ufunc(
                distance_matrix_, area_weighted_mean(ds[varn]),
                input_core_dims=[['model_ensemble']],
                output_core_dims=[['matrix_dim']],
                vectorize=True)

        else:
            # calculate a area-weighted condensed distance matrix
            rmse = xr.apply_ufunc(
                weighted_distance_matrix, ds[varn],
                input_core_dims=[['model_ensemble', 'lat', 'lon']],
                output_core_dims=[['matrix_dim']],
                kwargs={'lat': ds['lat'].data},  # NOTE: comment out for unweighted
                vectorize=True,
            )

        if 'month' in rmse.dims:
            rmse = rmse.mean('month')
        rmse /= np.median(rmse)
        rmse_list.append(rmse)

    rmse = np.mean(rmse_list, axis=0)
    distance_matrix = rmse

    def dend(linkage_type='ward'):
        linkage_matrix = linkage(distance_matrix, linkage_type, optimal_ordering=False)
        fig = plt.figure(figsize=(6, 7))
        fig.subplots_adjust(left=.22, top=1., bottom=.05, right=1.)
        dendrogram(linkage_matrix, labels=labels,
                   orientation='right',
                   count_sort='ascending',
                   link_color_func=lambda x: 'k',
        )

        lbls = plt.gca().get_ymajorticklabels()
        for lbl in lbls:
            lbl.set_color(label_colors[lbl.get_text()])
            lbl.set_fontweight('bold')
            # lbl.set_fontstyle(label_styles[lbl.get_text()])
            # if ensembles:
            #     lbl.set_fontsize(5)

        title = ' '.join([
            f'Dendrogram ({linkage_type}; {", ".join(varns)}) {period}',
            f'({season}) {region} {mip_str}'])
        # plt.gca().set_title(title)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().set_xticks([])

        plt.gca().set_xlabel('Generalised distance (1)')

        plt.savefig(os.path.join(PLOTPATH, 'figure5_pr_anom.png'), dpi=300)
        plt.savefig(os.path.join(SAVEPATH2, 'figure5_pr_anom.pdf'))
        plt.clf()

    dend('average')  #


if __name__ == '__main__':
    main()
