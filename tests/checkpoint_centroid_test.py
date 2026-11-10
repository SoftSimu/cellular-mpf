#!/usr/bin/env python3
# Cellular MPF
# Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Check redundant centroid validation near periodic and rounding boundaries."""

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from relaxed_2d_to_slab_centres import centroid_matches


class CentroidValidation(unittest.TestCase):
    def test_equivalent_arithmetic(self):
        self.assertTrue(centroid_matches(5.088176635581476e-7, 5.088177204015665e-7, 492))
        self.assertTrue(centroid_matches(9.763647312865942e-8, 9.763652997207828e-8, 492))
        self.assertTrue(centroid_matches(0., 492., 492))

    def test_one_float_rounding_unit(self):
        ulp = 2.**-16
        self.assertTrue(centroid_matches(128. + ulp, 128., 492))
        self.assertFalse(centroid_matches(128. + 2*ulp, 128., 492))

    def test_invalid_coordinates(self):
        for stored in (-1., 493., math.inf, math.nan, 1.e-5):
            self.assertFalse(centroid_matches(stored, 0., 492))
        self.assertFalse(centroid_matches(0., 0., 0))


if __name__ == '__main__':
    unittest.main()
