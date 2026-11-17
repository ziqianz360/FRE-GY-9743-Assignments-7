import unittest

import numpy as np

from fixedincomelib import qfCreate1DInterpolator
from fixedincomelib.utilities.numerics import (
    ExtrapMethod, Interpolator1DMonotoneConvex,
)


class MonotoneConvexTests(unittest.TestCase):
    def setUp(self):
        self.curve = qfCreate1DInterpolator(
            [1.0, 2.0, 4.0], [2.0, 3.0, 7.0], 'MONOTONE_CONVEX', 'FLAT',
        )

    def test_factory_selects_subclass_and_preserves_nodes(self):
        self.assertIsInstance(self.curve, Interpolator1DMonotoneConvex)
        for x, y in zip(self.curve.axis1, self.curve.values):
            self.assertEqual(self.curve.interpolate(x), y)
        self.assertEqual(self.curve.interpolate(1.5), 2.5)
        self.assertEqual(self.curve.interpolate(3.0), 5.0)

    def test_continuity_at_nodes_and_flat_extrapolation(self):
        for x, y in zip(self.curve.axis1, self.curve.values):
            self.assertAlmostEqual(self.curve.interpolate(x - 1e-9), y, places=8)
            self.assertAlmostEqual(self.curve.interpolate(x + 1e-9), y, places=8)
        self.assertEqual(self.curve.interpolate(-10.0), 2.0)
        self.assertEqual(self.curve.interpolate(10.0), 7.0)

    def test_shape_on_nonuniform_nodes(self):
        for values, direction in [([1.0, 2.0, 8.0, 20.0], 1),
                                  ([6.0, 4.0, 1.0, -1.0], -1),
                                  ([0.0, 0.0, 0.0, 0.0], 1)]:
            with self.subTest(values=values):
                curve = qfCreate1DInterpolator(
                    [0.0, 0.5, 2.0, 4.0], values, 'MONOTONE_CONVEX', 'FLAT',
                )
                grid = np.linspace(0.0, 4.0, 401)
                samples = np.array([curve.interpolate(x) for x in grid])
                self.assertTrue(np.all(direction * np.diff(samples) >= -1e-12))
                self.assertTrue(np.all(np.diff(samples, n=2) >= -1e-12))

    def test_exact_integrals_across_nodes_and_extrapolation(self):
        # Slopes are 1 and 2; full interval areas are 2.5 and 10.
        for start, end, expected in [(1.0, 4.0, 12.5), (1.5, 3.0, 5.375),
                                     (0.0, 5.0, 21.5), (-2.0, 0.0, 4.0),
                                     (5.0, 7.0, 14.0), (4.0, 1.0, -12.5),
                                     (2.0, 2.0, 0.0)]:
            with self.subTest(start=start, end=end):
                self.assertAlmostEqual(self.curve.integrate(start, end), expected)

    def test_integral_additivity_and_derivative(self):
        self.assertAlmostEqual(self.curve.integrate(0.0, 5.0),
                               self.curve.integrate(0.0, 2.5)
                               + self.curve.integrate(2.5, 5.0))
        for x in [0.5, 1.5, 2.0, 3.0, 4.0, 4.5]:
            epsilon = 1e-5
            derivative = self.curve.integrate(x - epsilon, x + epsilon) / (2 * epsilon)
            self.assertAlmostEqual(derivative, self.curve.interpolate(x), delta=1e-5)

    def test_one_node_funding_curve_is_constant_everywhere(self):
        curve = qfCreate1DInterpolator([1.0], [0.0005], 'MONOTONE_CONVEX', 'FLAT')
        for x in [-2.0, 0.0, 1.0, 3.0]:
            self.assertEqual(curve.interpolate(x), 0.0005)
        self.assertAlmostEqual(curve.integrate(-2.0, 3.0), 0.0025)
        self.assertAlmostEqual(curve.integrate(3.0, -2.0), -0.0025)

    def test_two_nodes_and_negative_ifr(self):
        curve = qfCreate1DInterpolator([0.0, 2.0], [-0.01, 0.01],
                                       'MONOTONE_CONVEX', 'FLAT')
        self.assertAlmostEqual(curve.interpolate(1.0), 0.0)
        self.assertAlmostEqual(curve.integrate(0.0, 2.0), 0.0)

    def test_collinear_decimal_data_allow_roundoff(self):
        curve = qfCreate1DInterpolator([0.0, 1.0, 2.0], [0.01, 0.02, 0.03],
                                       'MONOTONE_CONVEX', 'FLAT')
        self.assertAlmostEqual(curve.integrate(0.0, 2.0), 0.04)

    def test_rejects_incompatible_shapes(self):
        for values, message in [([1.0, 3.0, 2.0], 'monotone'),
                                 ([1.0, 3.0, 4.0], 'secant slopes'),
                                 ([4.0, 3.0, 1.0], 'secant slopes')]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    Interpolator1DMonotoneConvex([0.0, 1.0, 2.0], values, ExtrapMethod.FLAT)

    def test_rejects_invalid_nodes_and_extrapolation(self):
        for axis, values in [([], []), ([1.0], [1.0, 2.0]),
                              ([[1.0]], [[1.0]]), ([1.0, 1.0], [1.0, 2.0]),
                              ([2.0, 1.0], [1.0, 2.0]), ([np.nan], [1.0]),
                              ([1.0], [np.inf])]:
            with self.subTest(axis=axis, values=values):
                with self.assertRaises(ValueError):
                    Interpolator1DMonotoneConvex(axis, values, ExtrapMethod.FLAT)
        with self.assertRaisesRegex(ValueError, 'FLAT'):
            Interpolator1DMonotoneConvex([1.0], [1.0], ExtrapMethod.LINEAR)

    def test_rejects_nonfinite_queries(self):
        for x in [np.nan, np.inf, -np.inf]:
            with self.subTest(x=x):
                with self.assertRaises(ValueError):
                    self.curve.interpolate(x)
                with self.assertRaises(ValueError):
                    self.curve.integrate(x, 1.0)
                with self.assertRaises(ValueError):
                    self.curve.integrate(1.0, x)

    def test_optional_gradients_remain_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            self.curve.gradient_wrt_ordinate(2.0)
        with self.assertRaises(NotImplementedError):
            self.curve.gradient_of_integrated_value_wrt_ordinate(1.0, 4.0)


if __name__ == '__main__':
    unittest.main()
