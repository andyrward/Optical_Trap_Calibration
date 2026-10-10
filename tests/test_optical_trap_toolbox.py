import unittest
from pathlib import Path

import matplotlib
import numpy as np

from optical_trap_toolbox import (
    infer_experiment_segments,
    load_normalized_trace,
    plot_force_extension,
    plot_power_voltage,
    plot_segment_boundaries,
    plot_state_vs_time,
    segment_state_runs,
)


class OpticalTrapToolboxTestCase(unittest.TestCase):
    def test_statefinder_style_target_segmentation(self):
        states = np.array([0, 27, 27, 24, 24, 31, 31, 31, 24, 27], dtype=int)

        runs = segment_state_runs(states, state_code=27)

        self.assertEqual([(run.start_idx, run.end_idx, run.state_code) for run in runs], [(1, 2, 27), (9, 9, 27)])

    def test_full_run_segmentation_and_inference(self):
        states = np.array([21, 21, 24, 24, 27, 31, 3, 4], dtype=int)

        runs = segment_state_runs(states)
        inferred = infer_experiment_segments(states)

        self.assertEqual(len(runs), 6)
        self.assertEqual(len(inferred), 6)
        self.assertEqual(inferred[0].label, "tether_attempt_or_contact")
        self.assertEqual(inferred[2].label, "force_extension")
        self.assertEqual(inferred[3].label, "force_extension_subtether")
        self.assertEqual(inferred[4].label, "oscillator_mode_three")
        self.assertEqual(inferred[5].label, "oscillator_mode_four")

    def test_parse_matlab_sample_with_voltage_companion(self):
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "Matlab_Files" / "cal_250mw.dat"
        if not data_path.exists():
            self.skipTest(f"Sample file not available: {data_path}")

        trace = load_normalized_trace(data_path)

        expected_length = 32768
        self.assertEqual(trace["time_s"].shape[0], expected_length)
        self.assertEqual(trace["position_1_nm"].shape[0], expected_length)
        self.assertEqual(trace["position_2_nm"].shape[0], expected_length)
        self.assertTrue(np.isfinite(trace["power_mw"]).all())
        self.assertTrue(np.isfinite(trace["voltage_v"]).all())

    def test_plot_helpers_return_figures(self):
        matplotlib.use("Agg")
        time = np.arange(10, dtype=float)
        trace = {
            "time_s": time,
            "position_1_nm": np.linspace(0, 1, 10),
            "position_2_nm": np.linspace(1, 0, 10),
            "extension_nm": np.linspace(10, 20, 10),
            "force_pn": np.linspace(0, 5, 10),
            "voltage_v": np.linspace(4, 6, 10),
            "stiffness_pn_per_nm": np.full(10, np.nan),
            "state_code": np.array([21, 21, 24, 24, 27, 27, 31, 31, 3, 4], dtype=int),
            "power_mw": np.linspace(100, 120, 10),
        }
        segments = segment_state_runs(trace["state_code"])

        figs = [
            plot_state_vs_time(trace, segments),
            plot_power_voltage(trace),
            plot_force_extension(trace, segments),
            plot_segment_boundaries(trace, segments),
        ]

        for fig in figs:
            self.assertIsNotNone(fig)
            fig.clf()


if __name__ == "__main__":
    unittest.main()
