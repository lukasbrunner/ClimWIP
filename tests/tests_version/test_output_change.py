#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2018 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract:

"""
# import sys
import logging
import unittest
import numpy as np
import xarray as xr


class TestOutputChange(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @unittest.skip
    def test_output(self):
        ds = xr.open_dataset('./data/new/test_new.nc')
        ds_ref = xr.open_dataset('./data/reference/test_reference.nc')

        for varn in ds.variables:
            np.testing.assert_array_equal(
                ds[varn].data, ds_ref[varn].data, err_msg=varn)


if __name__ == '__main__':
    logging.disable(logging.NOTSET)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOutputChange)
    unittest.TextTestRunner(verbosity=2).run(suite)
