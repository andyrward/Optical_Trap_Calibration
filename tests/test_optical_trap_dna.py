import unittest

import numpy as np

from optical_trap_dna import (
    _fjc_extension_from_force,
    _langevin,
    _resolve_kbt,
    _wlc_extension_from_force,
    compare_delta_extension,
    delta_extension_target,
    delta_extension_zero_target,
    distance_estimate,
    full_length_curve,
    single_polymer_extension,
    target_curve,
    zero_target_curve,
)


class DNAForceExtensionTests(unittest.TestCase):
    def test_single_polymer_models_behavior(self):
        force = np.array([0.001, 0.002, 0.005, 0.01], dtype=float)
        x_wlc = single_polymer_extension(force, 1000.0, 50.0, 1200.0, model="wlc")
        self.assertTrue(np.all(np.diff(x_wlc) > 0.0))
        self.assertTrue(np.all(np.isfinite(x_wlc)))
        ratio = x_wlc / force
        self.assertTrue(np.allclose(ratio, ratio[0], rtol=0.08, atol=0.0))

        x_ext = single_polymer_extension(np.array([20.0, 50.0, 100.0]), 1000.0, 50.0, 1e6, model="extensible_wlc")
        self.assertAlmostEqual(np.max(x_ext), 985.75, delta=20.0)

    def test_extensible_models_stretch_past_contour_length(self):
        L = 4821.0
        force = np.array([40.0], dtype=float)
        lp = 50.0
        S = 1200.0
        x_wlc = single_polymer_extension(force, L, lp, S, model="extensible_wlc")
        x_fjc = single_polymer_extension(force, L, lp, S, model="extensible_fjc")
        self.assertGreater(float(x_wlc[0]), L)
        self.assertGreater(float(x_fjc[0]), L)

        odijk = L * (1.0 - 0.5 * np.sqrt(4.114 / (40.0 * lp)) + 40.0 / S)
        self.assertTrue(np.isclose(float(x_wlc[0]), odijk, rtol=0.01, atol=0.0))

    def test_fjc_matches_langevin_and_low_force_limit(self):
        L = 4821.0
        lp = 50.0
        force = np.array([0.1, 1.0, 5.0, 20.0], dtype=float)
        b = 2.0 * lp
        kbt = 4.114
        theoretical = L * (_langevin(force * b / kbt))
        x_fjc = _fjc_extension_from_force(force, L, lp, kbt)
        self.assertTrue(np.allclose(x_fjc, theoretical, rtol=1e-9, atol=1e-12))

        low_force = np.array([1e-8, 1e-5, 1e-4], dtype=float)
        expected_low = L * low_force * b / (3.0 * kbt)
        self.assertTrue(np.allclose(_fjc_extension_from_force(low_force, L, lp, kbt), expected_low, rtol=1e-5, atol=1e-12))

        x_ext = single_polymer_extension(force, L, lp, 1200.0, model="extensible_fjc")
        self.assertTrue(np.all(np.isfinite(x_ext)))

    def test_extensible_fjc_uses_multiplicative_smith_form(self):
        L = 1000.0
        lp = 0.75
        S = 50.0
        force = np.array([0.1, 1.0, 5.0, 20.0, 40.0], dtype=float)
        b = 2.0 * lp
        kbt = 4.114
        expected = L * _langevin(force * b / kbt) * (1.0 + force / S)
        x_ext = single_polymer_extension(force, L, lp, S, model="extensible_fjc")
        self.assertTrue(np.allclose(x_ext, expected, rtol=1e-9, atol=1e-12))

        additive = L * (_langevin(force * b / kbt) + force / S)
        gap = np.abs(additive - expected)
        self.assertGreater(float(np.max(gap)), 3.0)

    def test_temperature_and_base_count_inputs(self):
        self.assertAlmostEqual(_resolve_kbt(temperature_k=298.15), 4.114, delta=0.01)
        x_default = single_polymer_extension(np.array([1.0]), 1000.0, 50.0, 1200.0, model="wlc")
        x_temp = single_polymer_extension(np.array([1.0]), 1000.0, 50.0, 1200.0, model="wlc", temperature_k=298.15)
        self.assertTrue(np.allclose(x_default, x_temp, rtol=1e-3, atol=1e-6))

        bp = 13250
        contour = bp * 0.34
        self.assertAlmostEqual(contour, 4505.0, delta=1e-6)

        ds_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0, "rise_per_unit_nm": 0.34}
        force = np.array([1.0, 5.0])
        zero = zero_target_curve(force, bp, ds_params)
        expected = single_polymer_extension(force, contour, 50.0, 1200.0, model="wlc")
        self.assertTrue(np.allclose(zero, expected, rtol=1e-8, atol=1e-9))

        result = target_curve(force, bp, 300.0, 150.0, ds_params, ds_params, ds_params)
        expected_parallel = result["extension_nm"] - zero_target_curve(force, bp, ds_params)
        self.assertTrue(np.allclose(result["parallel_extension_nm"], expected_parallel, rtol=1e-8, atol=1e-9))

    def test_delta_distance_helpers(self):
        force = np.array([1.0, 5.0, 10.0], dtype=float)
        ds_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        target_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        loop_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        loop_length = 300.0

        full = full_length_curve(force, 1000.0, loop_length, ds_params, loop_params)
        zero = zero_target_curve(force, 1000.0, ds_params)
        target = target_curve(force, 1000.0, 300.0, loop_length, ds_params, target_params, loop_params)

        delta_0t = delta_extension_zero_target(force, 1000.0, loop_length, ds_params, loop_params)
        delta_target = delta_extension_target(force, 1000.0, 300.0, loop_length, ds_params, target_params, loop_params)
        dist = distance_estimate(force, 1000.0, 300.0, loop_length, ds_params, target_params, loop_params)
        self.assertTrue(np.allclose(delta_0t, full - zero, rtol=1e-8, atol=1e-9))
        self.assertTrue(np.allclose(delta_target, full - target["extension_nm"], rtol=1e-8, atol=1e-9))
        self.assertTrue(np.allclose(dist, target["extension_nm"] - zero, rtol=1e-8, atol=1e-9))

        compare = compare_delta_extension(delta_0t, force, delta_0t)
        self.assertEqual(compare["measured_delta_nm"].shape, force.shape)
        self.assertTrue(np.allclose(compare["residual_nm"], 0.0, atol=1e-12))

    def test_full_length_curve_reduces_to_single_dsDNA_curve(self):
        force = np.array([1.0, 5.0, 10.0, 50.0], dtype=float)
        ds_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        full = full_length_curve(force, 1000.0, 300.0, ds_params, ds_params)
        expected = single_polymer_extension(force, 1300.0, 50.0, 1200.0, model="wlc")
        self.assertTrue(np.allclose(full, expected, rtol=1e-8, atol=1e-9))

    def test_parallel_force_split_conserves_force_and_equal_extensions(self):
        force = np.array([5.0, 20.0], dtype=float)
        ds_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        target_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        loop_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}

        result = target_curve(force, 1000.0, 300.0, 150.0, ds_params, target_params, loop_params)
        self.assertTrue(np.allclose(result["force_target_pn"] + result["force_loop_pn"], force, rtol=1e-8, atol=1e-9))

        x_target = single_polymer_extension(result["force_target_pn"], 300.0, 50.0, 1200.0, model="wlc")
        x_loop = single_polymer_extension(result["force_loop_pn"], 150.0, 50.0, 1200.0, model="wlc")
        self.assertTrue(np.allclose(x_target, x_loop, rtol=1e-6, atol=1e-6))

    def test_zero_target_reference_uses_dsDNA_only(self):
        force = np.array([1.0, 5.0, 10.0], dtype=float)
        ds_params = {"model": "wlc", "persistence_length_nm": 50.0, "stretch_modulus_pn": 1200.0}
        zero = zero_target_curve(force, 1000.0, ds_params)
        expected = single_polymer_extension(force, 1000.0, 50.0, 1200.0, model="wlc")
        self.assertTrue(np.allclose(zero, expected, rtol=1e-8, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
