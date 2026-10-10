import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

from optical_trap_toolbox import (
    build_state_report,
    export_state_report_csv,
    export_state_report_json,
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
        self.assertEqual(inferred[1].label, "reference_pull")
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

    def test_parse_stateful_six_column_layout(self):
        data = np.array(
            [
                [0.0, 10.0, 2.0, 5.0, 0.3, 21.0],
                [1.0, 11.0, 3.0, 6.0, 0.4, 21.0],
                [2.0, 12.0, 4.0, 7.0, 0.5, 24.0],
                [3.0, 13.0, 5.0, 8.0, 0.6, 24.0],
                [4.0, 14.0, 6.0, 9.0, 0.7, 27.0],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stateful_6col.dat"
            np.savetxt(path, data, delimiter="\t")
            trace = load_normalized_trace(path)

        self.assertEqual(trace["extension_nm"].tolist(), [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(trace["force_pn"].tolist(), [2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(trace["voltage_v"].tolist(), [5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertEqual(trace["stiffness_pn_per_nm"].tolist(), [0.3, 0.4, 0.5, 0.6, 0.7])
        self.assertEqual(trace["state_code"].astype(int).tolist(), [21, 21, 24, 24, 27])

    def test_segment_state_runs_ignores_nan_values(self):
        states = np.array([np.nan, 21.0, 21.0, np.nan, 24.0, 24.0, np.nan], dtype=float)

        runs = segment_state_runs(states)

        self.assertEqual([(run.start_idx, run.end_idx, run.state_code) for run in runs], [(1, 2, 21), (4, 5, 24)])

    def test_state_report_build_and_exports(self):
        trace = {
            "time_s": np.linspace(0.0, 9.0, 10),
            "state_code": np.array([21, 21, 24, 24, 27, 27, 31, 31, 3, 4], dtype=int),
        }
        report = build_state_report(trace)

        self.assertEqual(report["total_states"], 6)
        self.assertEqual(report["total_runs"], 6)
        self.assertEqual(report["summary"][0]["label"], "oscillator_mode_three")
        self.assertEqual(report["summary"][0]["run_count"], 1)
        self.assertAlmostEqual(report["summary"][0]["total_duration_s"], 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "state_summary.csv"
            json_path = Path(temp_dir) / "state_summary.json"
            export_state_report_csv(report["summary"], csv_path)
            export_state_report_json(report["summary"], json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("state_code", csv_path.read_text(encoding="utf-8"))
            self.assertIn('"label": "tether_attempt_or_contact"', json_path.read_text(encoding="utf-8"))

    def test_state_report_on_repo_sample_file(self):
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "Matlab_Files" / "teterh1_target.dat"
        if not data_path.exists():
            self.skipTest(f"Sample file not available: {data_path}")

        trace = load_normalized_trace(data_path)
        report = build_state_report(trace)

        self.assertGreaterEqual(report["total_runs"], 1)
        self.assertGreater(len(report["state_runs"]), 0)
        self.assertIn("state_code", report["state_runs"][0])
        if report["summary"]:
            first_state = report["summary"][0]
            self.assertIn("label", first_state)
            self.assertIn("confidence", first_state)
            self.assertGreaterEqual(first_state["total_samples"], 1)

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
