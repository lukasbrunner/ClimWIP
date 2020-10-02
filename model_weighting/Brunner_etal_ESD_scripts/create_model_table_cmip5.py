#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2020 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import csv
import numpy as np
from model_weighting.core.get_filenames import get_filenames, select_variants

cfg = t = type('cfg', (), {})()

cfg.performance_diagnostics = ['tas', 'psl']
cfg.independence_diagnostics = None
cfg.target_diagnostic = None

cfg.subset = None
cfg.variants_use = 1
cfg.variants_select = 'natsorted'

cfg.model_id = ['CMIP5']
cfg.model_path = ['/net/atmos/data/cmip5-ng/']

cfg.model_scenario = ['rcp85']
filenames585 = get_filenames(cfg)['tas']

cfg.model_scenario = ['rcp26']
filenames126 = get_filenames(cfg)['tas']

common_model_ensembles = np.intersect1d([*filenames585.keys()], [*filenames126.keys()])

selected_models, selected_model_ensembles = select_variants(
    common_model_ensembles, 1, cfg.variants_select)

table = []
table.append(['Nr', 'Model', 'Variant'])
for nr, mv in enumerate(selected_model_ensembles):
    table.append(
        [nr+1, mv.split('_')[0], mv.split('_')[1]])


csv.register_dialect('unixpwd', delimiter=',')
with open('/home/lukbrunn/Documents/Projects/Paper_CMIP6/table_models_cmip5.csv', 'w') as ff:
    writer = csv.writer(ff, 'unixpwd')
    writer.writerows(table)
