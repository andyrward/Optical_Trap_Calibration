import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

from optical_trap_toolbox import (
    FORCE_EXTENSION_STATE,
    build_state_report,
    export_state_report_csv,
    export_state_report_json,
    infer_experiment_segments,
    load_normalized_trace,
    read_acquisition_params,
    plot_force_extension,
    plot_power_voltage,
    plot_segment_boundaries,
    plot_state_vs_time,
    segment_state_runs,
    summarize_pulls,
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
        self.assertEqual(inferred[1].label, "approach")       # state 24
        self.assertEqual(inferred[2].label, "measure_reference")  # state 27
        self.assertEqual(inferred[3].label, "force_extension_subtether")  # state 31
        self.assertEqual(inferred[4].label, "oscillator_hold")  # state 3
        self.assertEqual(inferred[5].label, "oscillator_ramp")  # state 4

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

    def test_params_parser_supports_new_and_legacy_formats(self):
        repo_root = Path(__file__).resolve().parents[1]

        tether_params = repo_root / "example_data" / "ParamData" / "tether1_params.dat"
        osc_params = repo_root / "example_data" / "ParamData" / "tether1_osc_params.dat"
        puppet_params = repo_root / "example_data" / "ParamData" / "tether1_puppet_params.dat"
        legacy_params = repo_root / "example_data" / "tether1_1_params.dat"

        tether = read_acquisition_params(tether_params)
        self.assertEqual(tether["mode"], "tether_formation")
        self.assertEqual(tether["tether_voltages"], [9.45, 8.4, 6.0, 5.0])

        osc = read_acquisition_params(osc_params)
        self.assertEqual(osc["mode"], "oscillator")
        self.assertEqual(osc["oscillator_voltages"], [8.4, 4.0])
        self.assertEqual(osc["oscillator_waits"], [1.0, 0.1])

        puppet = read_acquisition_params(puppet_params)
        self.assertEqual(puppet["mode"], "puppet")
        self.assertEqual(len(puppet["puppet_schedule"]), 5)
        self.assertEqual(puppet["puppet_schedule"][0]["target_v"], 8.4)

        legacy = read_acquisition_params(legacy_params)
        self.assertTrue(legacy["legacy"])
        self.assertEqual(legacy["mode_code"], 0)
        self.assertEqual(legacy["schedule_rows"], [])

        trace = load_normalized_trace(repo_root / "example_data" / "ParamData" / "tether1.dat")
        self.assertEqual(trace["acquisition_mode"][0], "tether_formation")
        self.assertTrue(np.all(trace["params_legacy"] == False))

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
        self.assertEqual(report["summary"][0]["label"], "oscillator_hold")  # state 3
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

    def test_parse_seven_column_primary_and_actuation_states(self):
        # 7-column layout: [t, t1a, t2a, t1b, t2b, primary_state, actuation_state]
        data = np.array(
            [
                [0.0, 5001.0, 18874.0, 8313.0, 16085.0, 7.0, 0.0],
                [1.0, 5002.0, 18875.0, 8314.0, 16086.0, 21.0, 3.0],
                [2.0, 5003.0, 18876.0, 8315.0, 16087.0, 27.0, 5.0],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "seven_col.dat"
            np.savetxt(path, data, delimiter=" ")
            trace = load_normalized_trace(path)

        np.testing.assert_array_equal(trace["state_code"].astype(int), [7, 21, 27])
        np.testing.assert_array_equal(trace["actuation_state_code"].astype(int), [0, 3, 5])
        # voltage/power should be NaN (no companion file present)
        self.assertTrue(np.all(np.isnan(trace["voltage_v"])))
        self.assertTrue(np.all(np.isnan(trace["power_mw"])))

    def test_force_extension_pull_count_and_intervals(self):
        # Build a trace: state 21, then 3 pulls (state 29) separated by state 24.
        # Pull durations span 3 samples; gaps span 2 samples.
        dt = 0.01
        states = np.array(
            [21, 21, 29, 29, 29, 24, 24, 29, 29, 29, 24, 29, 29, 29, 21],
            dtype=int,
        )
        n = len(states)
        time = np.arange(n, dtype=float) * dt
        trace = {"time_s": time, "state_code": states.astype(float)}

        result = summarize_pulls(trace, pull_state=FORCE_EXTENSION_STATE)

        self.assertEqual(result["n_pulls"], 3)
        self.assertEqual(len(result["pulls"]), 3)
        self.assertAlmostEqual(result["pulls"][0]["start_time_s"], 0.02)
        self.assertAlmostEqual(result["pulls"][0]["end_time_s"], 0.04)
        self.assertAlmostEqual(result["pulls"][0]["duration_s"], 0.02)
        self.assertAlmostEqual(result["pulls"][1]["start_time_s"], 0.07)
        self.assertAlmostEqual(result["pulls"][2]["start_time_s"], 0.11)
        self.assertEqual(len(result["gaps"]), 2)
        self.assertIn(24, result["gaps"][0]["states"])
        self.assertIn(24, result["gaps"][1]["states"])

    def test_build_state_report_includes_pulls(self):
        trace = {
            "time_s": np.linspace(0.0, 9.0, 10),
            "state_code": np.array([21, 21, 29, 29, 24, 24, 29, 29, 31, 31], dtype=float),
        }
        report = build_state_report(trace)
        self.assertIn("pulls", report)
        self.assertEqual(report["pulls"]["n_pulls"], 2)

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
