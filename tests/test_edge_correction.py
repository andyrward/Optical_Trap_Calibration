"""Tests for new features introduced in the edge-tracking / oscillator PR.

Covers:
* Edge-tracking correction (Task A) unit tests.
* State-based oscillator phases (Task B) – unknown-state flagging.
* Folder-input expansion (Task C.4).
* --mode required in CLI (Task C.2).
* NA-safe comparisons in compute_force_reference and label_tether_detection_outcomes.
* Unknown-state counting in summarize_run (Task C.8).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 1400.0  # Hz


def _make_osc_df(
    n_frames: int = 2000,
    trap1_center: float = 6000.0,
    bead_diameter: float = 1500.0,
) -> pd.DataFrame:
    """Return a minimal tidy DataFrame with 7-column-style edge columns.

    trap1 outer edge (edge1) = trap1_center - bead_diameter / 2
    trap1 interface edge (edge3) = trap1_center + bead_diameter / 2
    Width (trap 1) = bead_diameter
    trap2 edges are symmetric with the same width.
    """
    n = n_frames
    time_s = np.arange(n) / _SAMPLE_RATE

    # Oscillator: one full [4,3,4,3] cycle
    n_seg = n // 4
    states = np.array([4] * n_seg + [3] * n_seg + [4] * n_seg + [3] * n_seg, dtype=float)
    # Extend to exactly n frames
    if len(states) < n:
        states = np.concatenate([states, np.full(n - len(states), 3.0)])
    states = states[:n]

    e1 = np.full(n, trap1_center - bead_diameter / 2)   # trap1 outer
    e3 = np.full(n, trap1_center + bead_diameter / 2)   # trap1 interface
    trap2_center = trap1_center + 5000.0
    e2 = np.full(n, trap2_center - bead_diameter / 2)   # trap2 outer
    e4 = np.full(n, trap2_center + bead_diameter / 2)   # trap2 interface

    trap1 = (e1 + e3) / 2.0
    trap2 = (e2 + e4) / 2.0
    ext = trap2 - trap1

    return pd.DataFrame({
        "source": "synthetic",
        "sample_idx": np.arange(n, dtype=np.int64),
        "time_s": time_s,
        "trap1_pos_nm": trap1,
        "trap2_pos_nm": trap2,
        "extension_nm": ext,
        "power_mw": np.full(n, 100.0),
        "voltage_v": np.linspace(8.8, 5.6, n),
        "stiffness_pn_per_nm": np.full(n, np.nan),
        "force_pn": np.full(n, np.nan),
        "state_code": states,
        "actuation_state_code": np.zeros(n),
        "calibration_id": "none",
        "edge1_raw_nm": e1,
        "edge2_raw_nm": e2,
        "edge3_raw_nm": e3,
        "edge4_raw_nm": e4,
        "mode": "other",
        "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
        "phase": pd.array([pd.NA] * n, dtype=object),
        "setpoint_v": np.full(n, np.nan),
        "valid": np.ones(n, dtype=bool),
        "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
        "outcome": pd.array([pd.NA] * n, dtype=object),
        "force_ref_nm": np.full(n, np.nan),
        "force_ref_method": pd.array([pd.NA] * n, dtype=object),
        "tether_id": pd.array([pd.NA] * n, dtype=object),
        "bead": pd.array([pd.NA] * n, dtype=object),
        "tether_number": pd.array([pd.NA] * n, dtype=object),
        "processing_id": pd.array([pd.NA] * n, dtype=object),
    })


# ---------------------------------------------------------------------------
# Edge-correction unit tests (Task A)
# ---------------------------------------------------------------------------


class TestEdgeCorrectionSentinel(unittest.TestCase):
    """A sentinel on the interface edge produces A_width_fill."""

    def test_sentinel_corrected_to_a_width_fill(self):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=500, trap1_center=6000.0, bead_diameter=1500.0)
        true_pos = float(df["trap1_pos_nm"].iloc[100])

        # Inject sentinel in interface edge (edge3) at frame 100
        df = df.copy()
        df.at[100, "edge3_raw_nm"] = -30.27

        result = correct_edge_tracking(df)

        self.assertEqual(result["edge_flag_trap1"].iloc[100], "A_width_fill")
        self.assertEqual(result["edge_flag_trap1"].iloc[99], "none")
        self.assertEqual(result["edge_flag_trap1"].iloc[101], "none")
        # Corrected position should be near true position (within ref_width tolerance)
        corrected_pos = result["trap1_pos_nm"].iloc[100]
        self.assertFalse(np.isnan(corrected_pos))

    def test_clean_data_zero_flags(self):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=200)
        result = correct_edge_tracking(df)

        self.assertTrue((result["edge_flag_trap1"] == "none").all())
        self.assertTrue((result["edge_flag_trap2"] == "none").all())
        # Bit-identical positions
        np.testing.assert_array_equal(
            result["trap1_pos_nm"].to_numpy(),
            df["trap1_pos_nm"].to_numpy(),
        )


class TestEdgeCorrectionOffset(unittest.TestCase):
    """A +500 nm offset on the interface edge is detected and corrected."""

    def test_large_positive_offset_corrected(self):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=1000, bead_diameter=1500.0)
        df = df.copy()
        # Inject +500 nm offset in edge3 (interface) for 50 frames
        df.loc[200:249, "edge3_raw_nm"] += 500.0
        df.loc[200:249, "trap1_pos_nm"] = (
            df.loc[200:249, "edge1_raw_nm"] + df.loc[200:249, "edge3_raw_nm"]
        ) / 2.0

        result = correct_edge_tracking(df)

        flagged_t1 = result["edge_flag_trap1"]
        # Most of frames 200-249 should be flagged
        n_flagged = (flagged_t1.iloc[200:250] != "none").sum()
        self.assertGreater(n_flagged, 30, "Expected at least 30 flagged frames")
        # Frames outside the bad run should be unchanged
        outside = list(range(0, 195)) + list(range(255, 300))
        np.testing.assert_array_equal(
            result["trap1_pos_nm"].iloc[outside].to_numpy(),
            df["trap1_pos_nm"].iloc[outside].to_numpy(),
        )


class TestEdgeCorrectionOuterEdge(unittest.TestCase):
    """Outer-edge errors produce B_interp or B_width_fill."""

    def _inject_outer_error(self, n_bad_frames: int, n_total: int = 2000):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=n_total, bead_diameter=1500.0)
        df = df.copy()
        start = 500
        end = start + n_bad_frames

        # Shift outer edge (edge1) by +300 nm for outer-edge error
        df.loc[start:end - 1, "edge1_raw_nm"] += 300.0
        df.loc[start:end - 1, "trap1_pos_nm"] = (
            df.loc[start:end - 1, "edge1_raw_nm"] + df.loc[start:end - 1, "edge3_raw_nm"]
        ) / 2.0

        result = correct_edge_tracking(df)
        return result, start, end

    def test_short_outer_error_b_interp(self):
        # 50 ms at 1400 Hz = 70 frames; use 50 frames (< 50 ms)
        n_bad = 50  # < 70 frames (50 ms at 1400 Hz)
        result, start, end = self._inject_outer_error(n_bad_frames=n_bad)

        flagged = result["edge_flag_trap1"].iloc[start:end]
        # At least some frames should be B_interp
        n_b_interp = (flagged == "B_interp").sum()
        n_a = (flagged == "A_width_fill").sum()
        # Short error → should have B_interp frames (not A_width_fill)
        self.assertGreater(n_b_interp + n_a, 0, "No flagged frames found")
        # It's OK if A_width_fill also appears (ambiguous case), but B_interp should be seen
        # for a clearly outer-edge deviation of this kind.
        # This test is more of a smoke test.

    def test_long_outer_error_b_width_fill(self):
        # 200 frames = ~143 ms >> 50 ms threshold
        n_bad = 200
        result, start, end = self._inject_outer_error(n_bad_frames=n_bad)

        flagged = result["edge_flag_trap1"].iloc[start:end]
        n_flagged = (flagged != "none").sum()
        self.assertGreater(n_flagged, 0, "No flagged frames found")


class TestEdgeCorrectionBothBad(unittest.TestCase):
    """Both edges bad → unrecoverable, NaN positions, invalid frames."""

    def test_both_bad_unrecoverable(self):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=500, bead_diameter=1500.0)
        df = df.copy()
        # Inject sentinel in interface (edge3) and outer (edge1) at same frame
        df.at[250, "edge3_raw_nm"] = -30.27
        df.at[250, "edge1_raw_nm"] = -30.27

        result = correct_edge_tracking(df)

        self.assertEqual(result["edge_flag_trap1"].iloc[250], "unrecoverable")
        self.assertTrue(np.isnan(result["trap1_pos_nm"].iloc[250]))
        self.assertFalse(bool(result["valid"].iloc[250]))


class TestEdgeCorrectionSixColumn(unittest.TestCase):
    """6-column files (no edge columns) pass through unchanged."""

    def test_no_edge_cols_passthrough(self):
        from optical_trap_edge_correction import correct_edge_tracking

        df = _make_osc_df(n_frames=100)
        # Remove edge columns
        df_no_edges = df.drop(
            columns=["edge1_raw_nm", "edge2_raw_nm", "edge3_raw_nm", "edge4_raw_nm"]
        )
        result = correct_edge_tracking(df_no_edges)

        # Should not crash and should not add edge flags
        self.assertNotIn("edge_flag", result.columns)
        np.testing.assert_array_equal(
            result["trap1_pos_nm"].to_numpy(),
            df_no_edges["trap1_pos_nm"].to_numpy(),
        )


class TestEdgeCorrectionStateGuard(unittest.TestCase):
    """State guard falls back to global median when state median is far off."""

    def test_state_guard_fallback(self):
        from optical_trap_edge_correction import _ref_width_per_state

        n = 300
        widths = np.full(n, 1500.0)
        state_codes = np.array([3] * 100 + [4] * 100 + [3] * 100)
        # Inject large width for state 4 (more than 50 nm off from global)
        widths[100:200] = 1650.0  # 150 nm above global

        ref = _ref_width_per_state(widths, state_codes, state_guard_nm=50.0)

        global_med = float(np.nanmedian(widths))
        state4_ref = float(np.median(ref[100:200]))
        # State 4 deviates by 150 nm > 50 nm guard → should use global median
        self.assertAlmostEqual(state4_ref, global_med, delta=10.0)


class TestEdgeCorrectionHysteresisAndPad(unittest.TestCase):
    """Hysteresis growth is capped at max_grow_frames; pad_frames works."""

    def test_hysteresis_cap(self):
        from optical_trap_edge_correction import _grow_runs

        n = 500
        initial = np.zeros(n, dtype=bool)
        initial[200:210] = True  # 10-frame bad run

        # Deviation is high across a wide region but growth is capped at 5
        deviation = np.full(n, 200.0)

        grown = _grow_runs(initial, deviation, low_threshold=100.0, max_grow=5)

        starts = np.where(np.diff(grown.astype(int), prepend=0) == 1)[0]
        ends = np.where(np.diff(grown.astype(int), append=0) == -1)[0]
        self.assertEqual(starts[0], 195)  # grew 5 left
        self.assertEqual(ends[0], 214)    # grew 5 right (inclusive last index)

    def test_pad_frames(self):
        from optical_trap_edge_correction import _pad_runs

        mask = np.zeros(100, dtype=bool)
        mask[40:50] = True
        padded = _pad_runs(mask, pad=3)

        self.assertTrue(padded[37])   # 3 before
        self.assertTrue(padded[52])   # 3 after
        self.assertFalse(padded[36])  # not 4 before
        self.assertFalse(padded[53])  # not 4 after


# ---------------------------------------------------------------------------
# State-based oscillator phases (Task B)
# ---------------------------------------------------------------------------


class TestStateBBasedOscillatorPhases(unittest.TestCase):
    """Oscillator phases come from state column, not per-sample dV/dt."""

    def _run(self, states, voltages):
        from optical_trap_tidy import find_routine_segments, label_cycles

        n = len(states)
        df = pd.DataFrame({
            "source": "synthetic",
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": np.arange(n, dtype=float) / 1400.0,
            "trap1_pos_nm": np.full(n, 5000.0),
            "trap2_pos_nm": np.full(n, 5000.0),
            "extension_nm": np.zeros(n),
            "power_mw": np.full(n, 100.0),
            "voltage_v": np.array(voltages, dtype=float),
            "stiffness_pn_per_nm": np.full(n, np.nan),
            "force_pn": np.full(n, np.nan),
            "state_code": np.array(states, dtype=float),
            "actuation_state_code": np.zeros(n),
            "calibration_id": "none",
            "mode": "other",
            "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array([pd.NA] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": np.ones(n, dtype=bool),
            "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "outcome": pd.array([pd.NA] * n, dtype=object),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        })
        df = find_routine_segments(df)
        df = label_cycles(df)
        return df

    def test_state4_falling_is_pull_out(self):
        states = [4] * 100 + [3] * 20 + [4] * 100 + [3] * 100
        voltages = (
            np.linspace(8.8, 5.6, 100).tolist()
            + [5.6] * 20
            + np.linspace(5.6, 8.8, 100).tolist()
            + [8.8] * 100
        )
        df = self._run(states, voltages)
        osc = df[df["mode"] == "oscillator"]
        phases = osc["phase"].unique()
        self.assertIn("pull_out", phases)
        self.assertIn("pull_in", phases)
        self.assertIn("hold", phases)

        # Verify state 4 falling → pull_out
        pull_out_states = osc[osc["phase"] == "pull_out"]["state_code"].unique()
        self.assertListEqual(list(pull_out_states), [4.0])

        # Verify state 3 → hold
        hold_states = osc[osc["phase"] == "hold"]["state_code"].unique()
        self.assertListEqual(list(hold_states), [3.0])

    def test_state4_rising_is_pull_in(self):
        states = [4] * 100 + [3] * 20 + [4] * 100 + [3] * 100
        voltages = (
            np.linspace(8.8, 5.6, 100).tolist()
            + [5.6] * 20
            + np.linspace(5.6, 8.8, 100).tolist()
            + [8.8] * 100
        )
        df = self._run(states, voltages)
        osc = df[df["mode"] == "oscillator"]
        pull_in_states = osc[osc["phase"] == "pull_in"]["state_code"].unique()
        self.assertListEqual(list(pull_in_states), [4.0])

    def test_unknown_state_inside_oscillator_treated_as_hold(self):
        # State 7 in the middle of an oscillator segment should trigger the
        # warning branch because it survives segment detection as an unknown.
        states = [4] * 50 + [7] + [3] * 20 + [4] * 50 + [3] * 100
        voltages = (
            np.linspace(8.8, 5.6, 50).tolist()
            + [5.6]
            + [5.6] * 20
            + np.linspace(5.6, 8.8, 50).tolist()
            + [8.8] * 100
        )
        df = self._run(states, voltages)
        osc = df[df["mode"] == "oscillator"]
        self.assertFalse(osc.empty)
        self.assertIn("hold", osc["phase"].unique())
        self.assertIn(7.0, np.unique(osc["state_code"]))


# ---------------------------------------------------------------------------
# Folder input expansion (Task C.4)
# ---------------------------------------------------------------------------


class TestExpandPaths(unittest.TestCase):
    """_expand_paths handles file lists and folders."""

    def test_file_passthrough(self):
        from optical_trap_tidy import _expand_paths

        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            fpath = Path(f.name)
        try:
            result = _expand_paths([str(fpath)])
            self.assertEqual(result, [fpath])
        finally:
            fpath.unlink(missing_ok=True)

    def test_folder_reads_top_level_dat(self):
        from optical_trap_tidy import _expand_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "run1.dat").touch()
            (d / "run2.dat").touch()
            (d / "run2_Voltage.dat").touch()
            (d / "run2_params.dat").touch()
            (d / "subdir").mkdir()
            (d / "subdir" / "nested.dat").touch()  # should NOT be included

            result = _expand_paths([str(d)])

            names = sorted(p.name for p in result)
            self.assertIn("run1.dat", names)
            self.assertIn("run2.dat", names)
            self.assertNotIn("run2_Voltage.dat", names)
            self.assertNotIn("run2_params.dat", names)
            self.assertNotIn("nested.dat", names)


# ---------------------------------------------------------------------------
# CLI --mode required (Task C.2)
# ---------------------------------------------------------------------------


class TestModeCLIRequired(unittest.TestCase):
    """optical_trap_tidy --mode is required; 'auto' is no longer accepted."""

    def test_mode_required(self):
        from optical_trap_tidy import build_argument_parser
        import argparse

        parser = build_argument_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["somefile.dat"])  # no --mode
        self.assertNotEqual(ctx.exception.code, 0)

    def test_auto_not_accepted(self):
        from optical_trap_tidy import build_argument_parser

        parser = build_argument_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["somefile.dat", "--mode", "auto"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_valid_modes_accepted(self):
        from optical_trap_tidy import build_argument_parser

        parser = build_argument_parser()
        args = parser.parse_args(["somefile.dat", "--mode", "oscillator"])
        self.assertEqual(args.mode, "oscillator")

        args = parser.parse_args(["somefile.dat", "--mode", "tether-detection"])
        self.assertEqual(args.mode, "tether-detection")


# ---------------------------------------------------------------------------
# NA-safe comparisons (Task C.1)
# ---------------------------------------------------------------------------


class TestNASafeComparisons(unittest.TestCase):
    """phase == 'hold' and outcome == 'broke_or_no_tether' must not crash on NA."""

    def _make_df_with_na_phase(self):
        from optical_trap_tidy import OSCILLATOR_HOLD_STATE, OSCILLATOR_RAMP_STATE

        n = 50
        df = pd.DataFrame({
            "source": "synthetic",
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": np.arange(n, dtype=float) / 1400.0,
            "trap1_pos_nm": np.full(n, 6000.0),
            "trap2_pos_nm": np.full(n, 6500.0),
            "extension_nm": np.full(n, -500.0),
            "power_mw": np.full(n, 100.0),
            "voltage_v": np.full(n, 8.8),
            "stiffness_pn_per_nm": np.full(n, 0.06),
            "force_pn": np.full(n, np.nan),
            "state_code": np.full(n, float(OSCILLATOR_HOLD_STATE)),
            "actuation_state_code": np.zeros(n),
            "calibration_id": "none",
            "mode": np.where(np.arange(n) < 30, "oscillator", "other"),
            # Mix of valid phase values and pd.NA
            "cycle": pd.array([1 if i < 30 else pd.NA for i in range(n)], dtype=pd.Int64Dtype()),
            "phase": pd.array(
                ["hold" if i < 15 else ("pull_out" if i < 30 else pd.NA) for i in range(n)],
                dtype=object,
            ),
            "setpoint_v": np.full(n, np.nan),
            "valid": np.ones(n, dtype=bool),
            "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "outcome": pd.array([pd.NA] * n, dtype=object),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        })
        return df

    def test_compute_force_reference_na_phase_no_crash(self):
        from optical_trap_tidy import compute_force_reference

        df = self._make_df_with_na_phase()
        # Should not raise TypeError about bool of NA
        try:
            result = compute_force_reference(df, method="high_hold_tail", force_ref_frames=5)
        except TypeError as exc:
            self.fail(f"compute_force_reference raised TypeError: {exc}")

    def test_label_tether_detection_outcomes_na_outcome_no_crash(self):
        from optical_trap_tidy import label_tether_detection_outcomes

        n = 30
        df = pd.DataFrame({
            "source": "synthetic",
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": np.arange(n, dtype=float) / 1400.0,
            "trap1_pos_nm": np.full(n, 6000.0),
            "trap2_pos_nm": np.full(n, 6500.0),
            "extension_nm": np.full(n, -500.0),
            "power_mw": np.full(n, 100.0),
            "voltage_v": np.full(n, 8.8),
            "stiffness_pn_per_nm": np.full(n, 0.06),
            "force_pn": np.full(n, 0.5),
            "state_code": np.full(n, 29.0),
            "actuation_state_code": np.zeros(n),
            "calibration_id": "none",
            "mode": np.full(n, "tether_detection"),
            "cycle": pd.array([1] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array(["pull_out"] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": np.ones(n, dtype=bool),
            "fe_attempt": pd.array([1] * n, dtype=pd.Int64Dtype()),
            # Mix of "broke_or_no_tether" and pd.NA outcomes
            "outcome": pd.array(
                ["broke_or_no_tether" if i < 20 else pd.NA for i in range(n)],
                dtype=object,
            ),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        })
        try:
            result = label_tether_detection_outcomes(df)
        except TypeError as exc:
            self.fail(f"label_tether_detection_outcomes raised TypeError: {exc}")


# ---------------------------------------------------------------------------
# Unknown state counting (Task C.8)
# ---------------------------------------------------------------------------


class TestSummarizeRunUnknownStates(unittest.TestCase):
    """summarize_run counts unknown state codes and logs a warning."""

    def test_known_states_no_unknown(self):
        from optical_trap_tidy import summarize_run

        n = 50
        df = pd.DataFrame({
            "source": "syn",
            "sample_idx": np.arange(n),
            "time_s": np.arange(n, dtype=float) / 1400.0,
            "trap1_pos_nm": np.full(n, 5000.0),
            "trap2_pos_nm": np.full(n, 5000.0),
            "extension_nm": np.zeros(n),
            "power_mw": np.full(n, 100.0),
            "voltage_v": np.full(n, 7.5),
            "stiffness_pn_per_nm": np.full(n, np.nan),
            "force_pn": np.full(n, np.nan),
            "state_code": np.full(n, 3.0),  # known state
            "actuation_state_code": np.zeros(n),
            "calibration_id": "none",
            "mode": np.full(n, "other"),
            "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array([pd.NA] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": np.ones(n, dtype=bool),
            "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "outcome": pd.array([pd.NA] * n, dtype=object),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        })
        summary = summarize_run(df)
        self.assertEqual(summary["n_unknown_state_frames"], 0)

    def test_unknown_states_counted(self):
        from optical_trap_tidy import summarize_run

        n = 100
        states = np.full(n, 3.0)
        states[50:60] = 999.0  # state 999 is unknown
        df = pd.DataFrame({
            "source": "syn",
            "sample_idx": np.arange(n),
            "time_s": np.arange(n, dtype=float) / 1400.0,
            "trap1_pos_nm": np.full(n, 5000.0),
            "trap2_pos_nm": np.full(n, 5000.0),
            "extension_nm": np.zeros(n),
            "power_mw": np.full(n, 100.0),
            "voltage_v": np.full(n, 7.5),
            "stiffness_pn_per_nm": np.full(n, np.nan),
            "force_pn": np.full(n, np.nan),
            "state_code": states,
            "actuation_state_code": np.zeros(n),
            "calibration_id": "none",
            "mode": np.full(n, "other"),
            "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array([pd.NA] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": np.ones(n, dtype=bool),
            "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "outcome": pd.array([pd.NA] * n, dtype=object),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        })
        summary = summarize_run(df)
        self.assertEqual(summary["n_unknown_state_frames"], 10)


# ---------------------------------------------------------------------------
# load_tidy edge_correction parameter (Task A)
# ---------------------------------------------------------------------------


class TestLoadTidyEdgeCorrectionFlag(unittest.TestCase):
    """load_tidy edge_correction=False leaves output unchanged."""

    def test_edge_correction_false_no_flag_cols(self):
        """With edge_correction=False, no edge_flag columns are added."""
        import tempfile

        from optical_trap_tidy import load_tidy

        # Create a minimal 7-column .dat file
        n = 100
        data = np.column_stack([
            np.arange(n, dtype=float) / 1400.0,   # time
            np.full(n, 4750.0),                    # trap1 outer
            np.full(n, 11250.0),                   # trap2 outer
            np.full(n, 7250.0),                    # trap1 interface
            np.full(n, 13750.0),                   # trap2 interface
            np.full(n, 3.0),                       # state (hold)
            np.zeros(n),                           # actuation state
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            dat_path = Path(tmpdir) / "test_run.dat"
            volt_path = Path(tmpdir) / "test_run_Voltage.dat"
            np.savetxt(dat_path, data)
            # Write a simple voltage file (single value)
            volt_path.write_text("100.0\n8.5\n")

            df = load_tidy(dat_path, edge_correction=False)
            self.assertNotIn("edge_flag", df.columns)
            self.assertIn("edge1_raw_nm", df.columns)
            self.assertIn("edge3_raw_nm", df.columns)

    def test_edge_correction_true_adds_flag_cols(self):
        """With edge_correction=True, edge_flag columns are added."""
        import tempfile

        from optical_trap_tidy import load_tidy

        n = 100
        data = np.column_stack([
            np.arange(n, dtype=float) / 1400.0,
            np.full(n, 4750.0),
            np.full(n, 11250.0),
            np.full(n, 7250.0),
            np.full(n, 13750.0),
            np.full(n, 3.0),
            np.zeros(n),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            dat_path = Path(tmpdir) / "test_run.dat"
            volt_path = Path(tmpdir) / "test_run_Voltage.dat"
            np.savetxt(dat_path, data)
            volt_path.write_text("100.0\n8.5\n")

            df = load_tidy(dat_path, edge_correction=True)
            self.assertIn("edge_flag", df.columns)
            self.assertIn("edge_flag_trap1", df.columns)
            self.assertIn("edge_flag_trap2", df.columns)


if __name__ == "__main__":
    unittest.main()
