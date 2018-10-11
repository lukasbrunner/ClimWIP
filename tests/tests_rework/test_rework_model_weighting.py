#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
import sys
import shutil
import logging
import unittest
import numpy as np
import xarray as xr
from copy import copy
from tempfile import mkdtemp

from utils_python.get_filenames import Filenames
from utils_python.utils import read_config
sys.path.insert(0, '../../')
from model_weighting import (
    calc_target,
    calc_predictors,
    set_up_filenames,
    calc_sigmas)


class TestReworkModelWeighting(unittest.TestCase):

    def setUp(self):
        self.tmpdir = mkdtemp(dir='/net/h2o/climphys/tmp')
        self.cfg = read_config('rework', '../../configs/config_tests.ini')
        self.cfg.save_path = self.tmpdir
        self.fn = set_up_filenames(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skip
    def test_config(self):
        config_ref = dict(
            data_path='/net/atmos/data/cmip5-ng/',  # string
            scenario=['rcp85'],  # list of one string
            target_masko=True,  # bool
            predictor_diagnostics=['tas', 'pr', 'tasclt'],  # list of strings
            predictor_derived=[False, False, True],  # list of bools
            syear_eval=[1980, 1980, 1980],  # list of int
            syear_hist=1951,  # int
            obsdata=None,  # None
            sigma_s2=.1  # float
        )

        for key in config_ref.keys():
            if isinstance(config_ref[key], list):
                self.assertListEqual(getattr(self.cfg, key), config_ref[key],
                                     msg='Not matching values for key {}'.format(key))
            else:
                self.assertEqual(getattr(self.cfg, key), config_ref[key],
                                 msg='Not matching values for key {}'.format(key))
    @unittest.skip
    def test_set_up_filenames(self):
        self.assertListEqual(self.fn.get_variable_values('scenario'), ['rcp85'])
        self.assertListEqual(self.fn.get_variable_values('varn'), sorted(['tas', 'pr', 'clt']))
        models = self.fn.get_variable_values('model')
        filenames = self.fn.get_filenames(subset={'varn':'tas'})
        for varn in self.fn.get_variable_values('varn'):
            self.assertListEqual(self.fn.get_variable_values('model'), models)
            self.assertListEqual(self.fn.get_filenames(subset={'varn': varn}),
                                 [fn.replace('tas', varn) for fn in filenames])
        for model in models:
            ensembles = self.fn.get_variable_values('ensemble', subset={'model': model,
                                                                        'varn': 'tas'})
            for varn in self.fn.get_variable_values('varn'):
                self.assertListEqual(self.fn.get_variable_values(
                    'ensemble', subset={'model': model, 'varn': varn}), ensembles)
    @unittest.skip
    def test_calc_target(self):
        models = ['ACCESS1-0', 'CMCC-CESM', 'FGOALS-g2']  # list of random models to test
        for model in models:
            fn = copy(self.fn)
            fn.apply_filter(model=model)
            data = calc_target(fn, self.cfg).squeeze()
            ds_hist = xr.open_dataset(
                './data/tas_mon_{}_rcp85_r1i1p1_1951-2005_JJAMEAN_CLIM_EUR_3SREX.nc'.format(model))
            ds_future = xr.open_dataset(
                './data/tas_mon_{}_rcp85_r1i1p1_2031-2060_JJAMEAN_CLIM_EUR_3SREX.nc'.format(model))
            reference_data = ds_future['tas'].data.squeeze() - ds_hist['tas'].data.squeeze()
            mask = np.ma.masked_invalid(data).mask
            reference_mask = np.ma.masked_invalid(reference_data).mask

            # check if masked values fit
            np.testing.assert_array_equal(mask, reference_mask)
            # check if data fit
            np.testing.assert_array_almost_equal(data, reference_data, decimal=5)

            cfg = copy(self.cfg)
            cfg.target_diagnostic = 'pr'
            data = calc_target(fn, cfg).squeeze()
            ds_hist = xr.open_dataset(
                './data/pr_mon_{}_rcp85_r1i1p1_1951-2005_JJAMEAN_CLIM_EUR_3SREX.nc'.format(model))
            ds_future = xr.open_dataset(
                './data/pr_mon_{}_rcp85_r1i1p1_2031-2060_JJAMEAN_CLIM_EUR_3SREX.nc'.format(model))
            reference_data = ds_future['pr'].data.squeeze() - ds_hist['pr'].data.squeeze()
            mask = np.ma.masked_invalid(data).mask
            reference_mask = np.ma.masked_invalid(reference_data).mask

            np.testing.assert_array_equal(mask, reference_mask)
            # NOTE: the accuracy for precipitation is only 2 decimals!
            np.testing.assert_array_almost_equal(data, reference_data, decimal=2)
    @unittest.skip
    def test_calc_diagnostics(self):
        self.fn.apply_filter(model=['ACCESS1-0', 'CMCC-CESM', 'FGOALS-g2'])
        delta_q, delta_u, _, _ = calc_predictors(self.fn, self.cfg)

        ds = xr.open_dataset('./data/test_calc_diag.nc')
        delta_q_ref = ds['delta_q'].data
        delta_u_ref = ds['delta_u'].data

        np.testing.assert_array_almost_equal(delta_q, delta_q_ref, decimal=3)
        np.testing.assert_array_almost_equal(delta_u, delta_u_ref, decimal=3)

    def test_calc_sigmas(self):  # 'xx' as this test needs to come after the others
        self.fn.apply_filter(model=['ACCESS1-0', 'CMCC-CESM', 'FGOALS-g2'])
        targets = calc_target(self.fn, self.cfg).squeeze()
        delta_q, delta_u, lat, lon= calc_predictors(self.fn, self.cfg)
        weights_sigmas = calc_sigmas(targets, delta_u, lat, lon, self.fn, self.cfg, debug=True)
        ds = xr.open_dataset('./data/test_calc_sigma.nc')
        weights_sigmas_ref = ds['weights'].data
        np.testing.assert_array_almost_equal(weights_sigmas, weights_sigmas_ref.swapaxes(0, 1), decimal=3)


if __name__ == '__main__':
    logging.disable(logging.NOTSET)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReworkModelWeighting)
    unittest.TextTestRunner(verbosity=2).run(suite)
