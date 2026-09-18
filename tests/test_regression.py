"""Small dependency-light regression suite for the course implementation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from homography import (apply_homography, build_A, diagnose, homography_distance,
                        reprojection_error, solve_homography_8dof,
                        solve_homography_svd)
from preprocess import resize_corners, validate_quad
from warp import warp_bilinear, warp_nearest


class HomographyRegressionTests(unittest.TestCase):
    def setUp(self):
        self.src = np.array([[0.0, 0.0], [599.0, 0.0],
                             [599.0, 799.0], [0.0, 799.0]])
        self.H_true = np.array([[1.02, -0.13, 40.0],
                                [0.07, 1.05, 25.0],
                                [3.0e-4, -2.0e-4, 1.0]])
        self.dst = apply_homography(self.H_true, self.src)

    def test_four_point_exact_solution(self):
        H = solve_homography_svd(self.src, self.dst, normalize=True)
        rmse, _ = reprojection_error(H, self.src, self.dst)
        self.assertLess(rmse, 1e-9)
        self.assertLess(homography_distance(H, self.H_true), 1e-9)

    def test_solver_and_normalization_factors_are_separable(self):
        H_svd_raw = solve_homography_svd(self.src, self.dst, normalize=False)
        H_8_raw, info_raw = solve_homography_8dof(self.src, self.dst, normalize=False)
        H_svd_norm = solve_homography_svd(self.src, self.dst, normalize=True)
        H_8_norm, info_norm = solve_homography_8dof(self.src, self.dst, normalize=True)
        self.assertTrue(info_raw["consistent"])
        self.assertTrue(info_norm["consistent"])
        self.assertLess(homography_distance(H_svd_raw, H_8_raw), 1e-7)
        self.assertLess(homography_distance(H_svd_norm, H_8_norm), 1e-7)

    def test_h33_zero_counterexample(self):
        H0 = np.array([[1.0, 0.0, 0.3],
                       [0.0, 1.0, 0.2],
                       [1.0e-3, 1.0e-3, 0.0]])
        src = np.array([[10.0, 10.0], [110.0, 10.0],
                        [110.0, 90.0], [10.0, 90.0]])
        dst = apply_homography(H0, src)
        H = solve_homography_svd(src, dst, normalize=True)
        rmse, _ = reprojection_error(H, src, dst)
        _, info = solve_homography_8dof(src, dst, normalize=False)
        self.assertLess(rmse, 1e-8)
        self.assertFalse(info["consistent"])

    def test_diagnose_distinguishes_nullspace(self):
        rng = np.random.default_rng(7)
        info = diagnose(rng.normal(size=(8, 9)))
        self.assertEqual(info["nullity"], 1)
        self.assertTrue(np.isinf(info["condition_number_full"]))
        self.assertTrue(np.isfinite(info["condition_number_nonzero"]))


class GeometryRegressionTests(unittest.TestCase):
    def test_warp_identity_and_mask(self):
        image = np.arange(12, dtype=np.float64).reshape(3, 4)
        bilinear, mask = warp_bilinear(image, np.eye(3), 4, 3, fill=255)
        nearest, nearest_mask = warp_nearest(image, np.eye(3), 4, 3, fill=255)
        np.testing.assert_allclose(bilinear, image)
        np.testing.assert_allclose(nearest, image)
        self.assertTrue(mask.all())
        self.assertTrue(nearest_mask.all())

    def test_quad_and_resize_validation(self):
        good = np.array([[1, 1], [9, 1], [9, 7], [1, 7]], dtype=float)
        self.assertEqual(validate_quad(good), [])
        bad = good[[0, 1, 3, 2]]
        self.assertTrue(validate_quad(bad))
        resized = resize_corners(good, (10, 8), (20, 16))
        np.testing.assert_allclose(resized[0], [2.5, 2.5])
        np.testing.assert_allclose(resized[2], [18.5, 14.5])


if __name__ == "__main__":
    unittest.main()
