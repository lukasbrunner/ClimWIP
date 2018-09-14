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
import unittest
import numpy as np
from tempfile import mkdtemp

from utils_python.get_filenames import Filenames
sys.path.insert(0, '../../')
from model_weighting import calc_target




class TestUtils(unittest.TestCase):

    def setUp(self):
        self.tmpdir = mkdtemp(dir='/net/h2o/climphys/tmp')

        # set up a default pseudo-config to pass to calc_target
        cfg = lambda: None
        cfg.target_diagnostic = 'tas'
        cfg.target_type = 'change'
        cfg.target_masko = True
        cfg.target_season = 'JJA'
        cfg.target_agg = 'CLIM'
        cfg.save_path = self.tmpdir
        cfg.freq = 'mon'
        cfg.ensembles = True
        cfg.syear_fut = 2031
        cfg.eyear_fut = 2060
        cfg.syear_hist = 1951
        cfg.eyear_hist = 2005
        cfg.region = 'EUR_3SREX'
        cfg.overwrite = True
        self.cfg = cfg

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_tas(self):

        fn = Filenames(
            '{varn}/{varn}_{freq}_{model}_{scenario}_{ensemble}_g025.nc')
        fn.apply_filter(freq=self.cfg.freq, scenario='rcp85', varn='tas')
        models = fn.get_variable_values('model')
        if 'EC-EARTH' in models:
            models.remove('EC-EARTH')
        models = models[:5]
        fn.apply_filter(model=models)

        targets = calc_target(fn, self.cfg)

        # TODO: --- continue here ---





if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)
