#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import re
import numpy as np
import xarray as xr
from natsort import natsorted

import csv
csv.register_dialect('unixpwd', delimiter=',')

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_driver = '/home/lukbrunn/Documents/Scripts/cmip6_doi_table/chromedriver'
driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver)
driver.implicitly_wait(60)

FILENAME = '/net/h2o/climphys/lukbrunn/Data/ModelWeighting/CMIP6/tas_global_ssp585_050_81-00.nc'


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
    models = np.array([me.split('_')[0] for me in model_ensemble])
    models_dict = {}
    models_1var_map = {}
    for model in natsorted(np.unique(models)):
        idx = np.where(models == model)[0]
        models_dict[model] = natsorted([me.split('_')[1] for me in np.array(model_ensemble)[idx]])
        models_1var_map[model] = natsorted(np.array(model_ensemble)[idx])[0]
    return models_dict, models_1var_map


def get_doi_manual(url):
    dict_manual = {
        'https://furtherinfo.es-doc.org/CMIP6.UA.MCM-UA-1-0.ssp126.none.r1i1p1f2': 'N/A',
        'https://furtherinfo.es-doc.org/CMIP6.UA.MCM-UA-1-0.ssp585.none.r1i1p1f2': 'N/A',
        'https://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1': 'https://doi.org/10.22033/ESGF/CMIP6.881',
        'https://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.ssp126.none.r1i1p1f1': 'https://doi.org/10.22033/ESGF/CMIP6.898',
        'https://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-CM6-1-HR.historical.none.r1i1p1f2': 'https://doi.org/10.22033/ESGF/CMIP6.1385',
    }

    if url in dict_manual.keys():
        return dict_manual[url]
    else:
        print(f'Failed for {url}')
        return ''


def get_doi_from_url(url):
    try:
        driver.get(url)
        doi = driver.find_elements_by_link_text("View @ DKRZ")[1].get_attribute("href")
        # print(doi)
    except:
        doi = get_doi_manual(url)
    return doi


def get_models_from_file():
    ds = xr.open_dataset(FILENAME)
    return ds, *get_model_variants(ds['model_ensemble'].data)


def get_doi_table(ds, models, keys):
    table = {}
    table['Header'] = [
        'Nr',
        'Institution',
        'Model',
        '#Variants',
        'Variants',
        'DOI historical',
        # 'Further info url historical',
        'DOI SSP1-2.6',
        # 'Further info url SSP1-2.6',
        'DOI SSP5-8.5',
        # 'Further info url SSP5-8.5',
    ]

    for nr, model in enumerate(models):
        table[model] = [
            str(nr + 1),  # counter
            model,  # model name
            str(len(models[model])),  # number of ensemble members
            '; '.join(models[model])  # ensemble member IDs
        ]

        for experiment in ['historical', 'ssp126', 'ssp585']:
            fn = str(ds.sel(model_ensemble=keys[model])['filename'].data)
            fn = fn.replace('ssp585', experiment)
            attrs = xr.open_dataset(fn).attrs
            url = attrs['further_info_url'].strip()
            # print(url)
            doi = get_doi_from_url(url)

            table[model].append(doi)  # doi
            # table[model].append(url)  # further info url

        # insert the institution name to the second column!
        institution = attrs['institution_id'].strip()
        table[model] = [table[model][0], institution, *table[model][1:]]

    table['Total'] = [
        '', '', '',
        str(len(ds['model_ensemble'])),
        '', '', '', '']

    return table


def get_last_change_dates(ds, varns=['tas', 'psl']):
    table = [
        ['model',
         'ensemble member',
         'variable',
         'experiment',
         'file name',
         # 'creation date',
         'version date',
         # 'archive date',
         'sha256',
         'tracking ID',
    ]]

    model_ensembles = ds['model_ensemble'].data
    for model_ensemble in model_ensembles:
        for varn in varns:
            fn_cmip6_ng = str(ds.sel(model_ensemble=model_ensemble)['filename'].data)
            fn_cmip6_ng = fn_cmip6_ng.replace('/tas/', f'/{varn}/').replace('tas_', f'{varn}_')
            ds_cmip6_ng = xr.open_dataset(fn_cmip6_ng)
            cmip6_raw_filenames = ds_cmip6_ng.attrs['original_file_names'].split(',')
            for fn in cmip6_raw_filenames:
                path, fn = os.path.split(fn.strip())
                fn_info = os.path.join(path, '.' + fn.replace('.nc', '.info'))
                with open(fn_info, 'r') as ff:
                    info = ff.read().splitlines()[-1]
                    creation_date, version_date, archive_date, sha_sum, tracking_id, _ = info.split()

                    table.append([
                        model_ensemble.split('_')[0],
                        model_ensemble.split('_')[1],
                        varn,
                        fn.split('_')[3],
                        fn,
                        # creation_date[:8],
                        version_date[:8],
                        # archive_date[:8],
                        sha_sum,
                        tracking_id,
                    ])

    return table


def main():
    ds, models, keys = get_models_from_file()

    doi_table = get_doi_table(ds, models, keys)
    with open('/home/lukbrunn/Documents/Projects/Paper_CMIP6/table_models_cmip6.csv', 'w') as ff:
        writer = csv.writer(ff, 'unixpwd')
        writer.writerows(doi_table.values())

    change_table = get_last_change_dates(ds)
    with open('/home/lukbrunn/Documents/Projects/Paper_CMIP6/table_models_cmip6_change.csv', 'w') as ff:
        writer = csv.writer(ff, 'unixpwd')
        writer.writerows(change_table)



if __name__ == '__main__':
    main()
