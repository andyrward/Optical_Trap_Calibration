import unittest

import numpy as np

from optical_trap_dna import (
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
