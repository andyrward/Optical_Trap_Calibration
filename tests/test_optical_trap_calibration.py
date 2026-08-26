import unittest
from pathlib import Path

from optical_trap_calibration import calibrate_file


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


if __name__ == "__main__":
    unittest.main()
