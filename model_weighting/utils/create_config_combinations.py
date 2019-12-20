#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-09-12 08:59:00 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os
import numpy as np

nr = 15

PATH = '../configs/eucp_tier1'
FILE = f'tas_combinations_{nr}.ini'

models = ['ACCESS1-0_r1i1p1_CMIP5', 'BNU-ESM_r1i1p1_CMIP5',
          'CCSM4_r1i1p1_CMIP5', 'CESM1-BGC_r1i1p1_CMIP5',
          'CMCC-CESM_r1i1p1_CMIP5', ' CNRM-CM5_r10i1p1_CMIP5',
          'CSIRO-Mk3-6-0_r10i1p1_CMIP5', 'CanESM2_r1i1p1_CMIP5',
          'FGOALS-g2_r1i1p1_CMIP5', 'GFDL-CM3_r1i1p1_CMIP5',
          'GISS-E2-H-CC_r1i1p1_CMIP5', 'HadGEM2-CC_r1i1p1_CMIP5',
          'IPSL-CM5A-LR_r1i1p1_CMIP5', ' MIROC-ESM-CHEM_r1i1p1_CMIP5',
          'MPI-ESM-LR_r1i1p1_CMIP5', 'MRI-CGCM3_r1i1p1_CMIP5',
          'NorESM1-ME_r1i1p1_CMIP5', ' bcc-csm1-1-m_r1i1p1_CMIP5',
          'inmcm4_r1i1p1_CMIP5']

with open(os.path.join(PATH, FILE), 'a') as ff:
    for idx in range(20):
        subset = np.random.choice(models, nr, replace=False)
        ff.write('\n')
        ff.write(f'[{FILE.replace(".ini", "")}_{idx}]\n')
        ff.write('subset = {}\n'.format(', '.join(subset)))
