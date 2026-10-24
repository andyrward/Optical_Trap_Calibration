import unittest
from pathlib import Path

import numpy as np

from filtpsdaliased import filtpsdaliased
from optical_trap_calibration import calibrate_file, load_position_data, load_voltage_file


class CalibrationTestCase(unittest.TestCase):
    def test_filtpsdaliased_matches_matlab_aliasing_term(self):
        x = np.array([5e-5, 110.0], dtype=float)
        xdata = np.array([10.0, 50.0, 250.0], dtype=float)

        expected = np.zeros_like(xdata, dtype=float)
        kt = 4.1
        et = 0.650e-3
        f_nyquist = 2.0 * xdata[-1]

        for i, fi in enumerate(xdata):
            total = 0.0
            for j in range(-40, 41):
                freq_term = fi + (j - 1) * f_nyquist
                denominator = (np.pi ** 2) * x[0] * ((freq_term ** 2) + (x[1] ** 2))
                total += (kt / denominator) * (np.sinc(freq_term * et)) ** 2
            expected[i] = total

        np.testing.assert_allclose(filtpsdaliased(x, xdata), expected)

    def test_example_dataset_calibrates(self):
        repo_root = Path(__file__).resolve().parents[1]
        example_file = repo_root / "example_data" / "example_trap_data.dat"

        results, trap_power = calibrate_file(example_file)

        self.assertTrue(results)
        self.assertGreater(trap_power, 0)
        for result in results:
            self.assertGreater(result["gamma"], 0)
            self.assertGreater(result["corner_frequency_hz"], 0)
            self.assertGreater(result["stiffness_pn_per_nm"], 0)
            self.assertGreater(result["stiffness_per_mw"], 0)

    def test_matlab_style_voltage_and_data_files_load(self):
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "example_data" / "Cal1_100mW.dat"
        voltage_path = repo_root / "example_data" / "Cal1_100mW_Voltage.dat"

        self.assertAlmostEqual(load_voltage_file(voltage_path), 102.1)

        tracks = load_position_data(data_path)
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            self.assertEqual(track.shape[0], 65536)

        results, trap_power = calibrate_file(data_path)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(trap_power, 102.1)
        for result in results:
            self.assertGreater(result["gamma"], 0)
            self.assertGreater(result["corner_frequency_hz"], 0)
            self.assertGreater(result["stiffness_pn_per_nm"], 0)
            self.assertGreater(result["stiffness_per_mw"], 0)


if __name__ == "__main__":
    unittest.main()
