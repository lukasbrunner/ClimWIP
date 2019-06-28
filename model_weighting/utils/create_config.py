#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-stamp: <2019-06-28 08:37:33 lukbrunn>

(c) 2019 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import os

PATH = '../configs/'
FILE = 'points_tas.ini'
NR_LATS = 19
NR_LONS = 20

with open(os.path.join(PATH, FILE), 'a') as ff:
    for idx_lat in range(NR_LATS):
        for idx_lon in range(NR_LONS):
            ff.write('\n')
            ff.write(f'[{FILE.replace(".ini", "")}_{idx_lat}-{idx_lon}]\n')
            ff.write(f'idx_lats = {idx_lat}\n')
            ff.write(f'idx_lons = {idx_lon}\n')
