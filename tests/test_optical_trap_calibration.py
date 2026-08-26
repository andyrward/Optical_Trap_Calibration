import unittest
from pathlib import Path

from optical_trap_calibration import calibrate_file, load_position_data, load_voltage_file


class CalibrationTestCase(unittest.TestCase):
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
            self.assertTrue(track.size > 0)

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
