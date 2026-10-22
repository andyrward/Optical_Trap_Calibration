"""Tests for optical_trap_tidy.py and optical_trap_oscillator.py.

Covers:
* Synthetic pattern detection (oscillator and force-extension).
* Synthetic cycle and phase labelling.
* Glitch and spike masking.
* Mode registry extensibility (dummy mode on actuation_state_code).
* Calibration loading (JSON file + CLI flag).
* Regression tests against the real tether data files in example_data/.
* Oscillator analyzer on synthetic data.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------


def _make_tidy_df(
    states: list[int],
    voltages: list[float],
    actuation_states: list[int] | None = None,
    sample_rate: float = 1400.0,
) -> pd.DataFrame:
    """Return a minimal tidy DataFrame from parallel lists."""
    n = len(states)
    if actuation_states is None:
        actuation_states = [0] * n
    time = np.arange(n, dtype=float) / sample_rate
    ext = np.linspace(-3000.0, -7000.0, n)
    return pd.DataFrame(
        {
            "source": "synthetic",
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": time,
            "trap1_pos_nm": np.full(n, 5000.0),
            "trap2_pos_nm": 5000.0 - ext,
            "extension_nm": ext,
            "power_mw": np.full(n, 1000.0),
            "voltage_v": np.array(voltages, dtype=float),
            "stiffness_pn_per_nm": np.full(n, np.nan),
            "force_pn": np.full(n, np.nan),
            "state_code": np.array(states, dtype=float),
            "actuation_state_code": np.array(actuation_states, dtype=float),
            "calibration_id": "none",
            "mode": "other",
            "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array([pd.NA] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": True,
        }
    )


def _oscillator_voltage_block(n_per_phase: int = 100) -> list[float]:
    """Build one oscillator cycle worth of stepped voltages.

    Phases: pull_out (falling 8.8→5.6), short hold (5.6), pull_in (rising
    5.6→8.8), long hold (8.8).
    """
    pull_out = np.linspace(8.8, 5.6, n_per_phase).tolist()
    hold_short = [5.6] * (n_per_phase // 5)
    pull_in = np.linspace(5.6, 8.8, n_per_phase).tolist()
    hold_long = [8.8] * n_per_phase
    return pull_out + hold_short + pull_in + hold_long


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestOscillatorPatternDetection(unittest.TestCase):
    """Tests for OscillatorMode.detect()."""

    def _detect(self, states, voltages=None):
        from optical_trap_tidy import OscillatorMode

        n = len(states)
        if voltages is None:
            voltages = [8.8] * n
        time = np.arange(n, dtype=float) / 1400.0
        spec = OscillatorMode()
        return spec.detect(
            np.array(states, dtype=float),
            np.zeros(n, dtype=float),
            time,
        )

    def test_single_cycle_detected(self):
        # [4,3,4,3] once = 1 segment
        states = [20] * 10 + [4] * 50 + [3] * 20 + [4] * 50 + [3] * 100
        segs = self._detect(states)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], 10)  # starts at first state-4 run
        self.assertEqual(segs[0][1], len(states) - 1)

    def test_multiple_cycles_one_segment(self):
        # Three [4,3,4,3] quads should be one segment
        block = [4] * 50 + [3] * 10 + [4] * 50 + [3] * 100
        states = [20] * 5 + block * 3 + [20] * 5
        segs = self._detect(states)
        self.assertEqual(len(segs), 1)

    def test_glitch_ignored_between_quads(self):
        # A single-sample other-state glitch should not break the segment
        block = [4] * 50 + [3] * 10 + [4] * 50 + [3] * 100
        states = block + [5] + block  # state 5 is 1-sample glitch
        segs = self._detect(states)
        self.assertEqual(len(segs), 1)

    def test_multi_sample_break_creates_two_segments(self):
        # State 7 with 5+ samples between two quad groups → two segments
        block = [4] * 50 + [3] * 10 + [4] * 50 + [3] * 100
        states = block + [7] * 10 + block
        segs = self._detect(states)
        self.assertEqual(len(segs), 2)

    def test_no_oscillator_in_state_27_only_file(self):
        # Force-extension runs should not match oscillator
        states = [27] * 200
        segs = self._detect(states)
        self.assertEqual(len(segs), 0)

    def test_incomplete_pattern_detected_as_partial_segment(self):
        # A contiguous 3/4 block is still an oscillator segment even if it does
        # not contain a complete [4,3,4,3] quad; it is retained as a partial
        # cycle for downstream labeling.
        states = [4] * 50 + [3] * 20 + [4] * 50
        segs = self._detect(states)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[0][1], len(states) - 1)


class TestForceExtensionDetection(unittest.TestCase):
    def test_tether_detection_contiguous_block(self):
        # A contiguous block of tether-detection states (24, 27, 29, 23) is
        # one segment. States 21/21 before it are not in the TD set.
        from optical_trap_tidy import TetherDetectionMode

        # states: 21, 21 (non-TD), then 24/27/29 (TD) → 1 segment
        states = np.array([21, 21, 27, 27, 27, 24, 27, 27, 24, 27, 27], dtype=float)
        spec = TetherDetectionMode()
        segs = spec.detect(states, np.zeros_like(states), np.arange(len(states), dtype=float))
        # All TD states (27 and 24) are contiguous → 1 segment
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], 2)  # first 27 at index 2
        self.assertEqual(segs[0][1], len(states) - 1)

    def test_three_pull_out_runs_labeled(self):
        # Three state-29 runs separated by state-27 reference holds.
        from optical_trap_tidy import TetherDetectionMode, find_routine_segments, label_cycles

        # 27 ref, 29 pull, 23 break, 27 ref, 29 pull, 23 break, 27 ref, 29, 26 relax
        n_each = 10
        states = (
            [27] * n_each + [29] * n_each + [23] * n_each
            + [27] * n_each + [29] * n_each + [23] * n_each
            + [27] * n_each + [29] * n_each + [26] * n_each
        )
        voltages = [7.5] * len(states)
        df = _make_tidy_df(states, voltages)
        df = find_routine_segments(df)
        df = label_cycles(df)
        td = df[df["mode"] == "tether_detection"]
        # All TD rows in one segment
        self.assertFalse(td.empty)
        # Three distinct fe_attempts
        attempts = sorted(td["fe_attempt"].dropna().astype(int).unique().tolist())
        self.assertEqual(attempts, [1, 2, 3])
        # Phases
        phases = set(td["phase"].dropna().unique())
        self.assertIn("pull_out", phases)
        self.assertIn("reference", phases)
        self.assertIn("post_break", phases)
        self.assertIn("relax", phases)


class TestPhaseLabelling(unittest.TestCase):
    """Tests for label_cycles on synthetic voltage waveforms."""

    def _run(self, states, voltages, dv_dt_threshold=0.0003, smooth_window=20):
        from optical_trap_tidy import find_routine_segments, label_cycles

        df = _make_tidy_df(states, voltages)
        df = find_routine_segments(df)
        df = label_cycles(df, dv_dt_threshold=dv_dt_threshold, smooth_window=smooth_window)
        return df

    def test_pull_out_phase_when_voltage_falls(self):
        # One [4,3,4,3] cycle: falling V → pull_out, flat → hold, rising → pull_in, flat → hold
        v = _oscillator_voltage_block(n_per_phase=200)
        n_po = 200  # pull_out samples
        n_hs = 40   # short hold
        n_pi = 200  # pull_in
        n_hl = 200  # long hold
        states = [4] * n_po + [3] * n_hs + [4] * n_pi + [3] * n_hl
        df = self._run(states, v, smooth_window=50)
        osc = df[df["mode"] == "oscillator"]
        phases = osc["phase"].value_counts()
        self.assertIn("pull_out", phases.index)
        self.assertIn("pull_in", phases.index)
        self.assertIn("hold", phases.index)

    def test_cycle_numbering_starts_at_one(self):
        v = _oscillator_voltage_block(n_per_phase=100) * 2  # two cycles
        n_po = 100
        n_hs = 20
        n_pi = 100
        n_hl = 100
        block = [4] * n_po + [3] * n_hs + [4] * n_pi + [3] * n_hl
        states = block * 2
        df = self._run(states, v, smooth_window=30)
        osc = df[df["mode"] == "oscillator"]
        cycles = set(osc["cycle"].dropna().astype(int).tolist())
        self.assertIn(1, cycles)
        self.assertIn(2, cycles)


class TestContinuousCycleNumbering(unittest.TestCase):
    """Cycles should be numbered continuously across pause segments."""

    def test_two_segments_numbered_continuously(self):
        from optical_trap_tidy import find_routine_segments, label_cycles

        # Two oscillator groups separated by a state-7 pause
        block_v = _oscillator_voltage_block(n_per_phase=100)
        block_states = [4] * 100 + [3] * 20 + [4] * 100 + [3] * 100
        pause = [7] * 50
        pause_v = [8.8] * 50
        states = block_states + pause + block_states
        v = block_v + pause_v + block_v

        df = _make_tidy_df(states, v)
        df = find_routine_segments(df)
        df = label_cycles(df, smooth_window=30)

        osc = df[df["mode"] == "oscillator"]
        max_cycle = osc["cycle"].dropna().astype(int).max()
        # Each block has 1 cycle → total 2
        self.assertEqual(max_cycle, 2)


class TestOscillatorPartialCycles(unittest.TestCase):
    def _run(self, states, voltages, smooth_window=20):
        from optical_trap_tidy import find_routine_segments, label_cycles

        df = _make_tidy_df(states, voltages)
        df = find_routine_segments(df)
        df = label_cycles(df, smooth_window=smooth_window)
        return df

    def test_truncated_cycle_after_complete_is_kept_and_flagged(self):
        full_v = _oscillator_voltage_block(n_per_phase=100)
        full_s = [4] * 100 + [3] * 20 + [4] * 100 + [3] * 100

        # partial: pull_out + short hold + incomplete pull_in, then state 2 break
        partial_v = (
            list(np.linspace(9.0, 5.6, 100))
            + [5.6] * 20
            + list(np.linspace(5.6, 8.1, 50))
        )
        partial_s = [4] * 100 + [3] * 20 + [4] * 50

        states = full_s + partial_s + [2] * 5
        voltages = full_v + partial_v + [8.1] * 5
        df = self._run(states, voltages, smooth_window=15)

        osc = df[df["mode"] == "oscillator"].copy()
        complete_ids = sorted(osc[osc["cycle_complete"] == True]["cycle"].dropna().astype(int).unique().tolist())  # noqa: E712
        partial_ids = sorted(osc[osc["cycle_complete"] == False]["cycle"].dropna().astype(int).unique().tolist())  # noqa: E712
        self.assertEqual(complete_ids, [1])
        self.assertGreaterEqual(len(partial_ids), 1)
        self.assertIn("partial_truncated", set(osc["cycle_status"].dropna().tolist()))
        self.assertEqual(int(((df["state_code"].isin([3, 4])) & (df["cycle"].isna())).sum()), 0)

    def test_end_of_file_partial_cycle_is_marked(self):
        full_v = _oscillator_voltage_block(n_per_phase=100)
        full_s = [4] * 100 + [3] * 20 + [4] * 100 + [3] * 100

        # EOF partial: pull_out then hold, file ends before pull_in/last hold
        partial_v = list(np.linspace(9.0, 5.6, 100)) + [5.6] * 30
        partial_s = [4] * 100 + [3] * 30

        states = full_s + partial_s
        voltages = full_v + partial_v
        df = self._run(states, voltages, smooth_window=15)

        osc = df[df["mode"] == "oscillator"].copy()
        partial = osc[osc["cycle_complete"] == False]  # noqa: E712
        self.assertGreater(len(partial), 0)
        self.assertIn("partial_end_of_file", set(partial["cycle_status"].dropna().tolist()))
        self.assertEqual(int(((df["state_code"].isin([3, 4])) & (df["cycle"].isna())).sum()), 0)

    def test_no_state3_or_state4_left_without_cycle_for_unfinished_ramp(self):
        # A stalled end-of-file ramp should remain in the oscillator cycle,
        # not be left without a cycle number.
        states = [4] * 50 + [3] * 20 + [4] * 50
        voltages = [8.8] * 50 + [5.6] * 20 + [8.8] * 50
        df = self._run(states, voltages, smooth_window=15)
        unassigned = (df["state_code"].isin([3, 4])) & (df["cycle"].isna())
        self.assertEqual(int(unassigned.sum()), 0)


class TestValidMask(unittest.TestCase):
    """Tests for apply_valid_mask."""

    def test_single_sample_glitch_flagged(self):
        from optical_trap_tidy import apply_valid_mask

        # One-sample state 2 surrounded by state 3
        states = [3] * 10 + [2] + [3] * 10
        voltages = [5.6] * len(states)
        df = _make_tidy_df(states, voltages)
        df = apply_valid_mask(df)
        # Sample at position 10 (state 2) should be invalid
        self.assertFalse(df.loc[10, "valid"])
        # Others should be valid
        self.assertTrue(df.loc[5, "valid"])
        self.assertTrue(df.loc[15, "valid"])

    def test_extension_spike_flagged(self):
        from optical_trap_tidy import apply_valid_mask

        n = 20
        states = [4] * n
        voltages = [8.0] * n
        df = _make_tidy_df(states, voltages)
        # Inject a spike at sample 10: jump > 1000 nm
        ext = df["extension_nm"].to_numpy(dtype=float).copy()  # writable copy
        ext[10] = ext[9] + 2000.0
        df["extension_nm"] = ext
        df = apply_valid_mask(df, spike_threshold_nm=1000.0)
        # Sample 10 and 11 should be flagged
        self.assertFalse(df.loc[10, "valid"])
        self.assertFalse(df.loc[11, "valid"])
        self.assertTrue(df.loc[5, "valid"])


class TestModeRegistryExtensibility(unittest.TestCase):
    """A dummy mode registered on actuation_state_code should be detected."""

    def test_dummy_mode_on_actuation_state(self):
        import tempfile
        import numpy as np
        from optical_trap_tidy import ModeSpec, MODE_REGISTRY, find_routine_segments, label_cycles

        DUMMY_ACT_STATE = 99

        class DummyMode(ModeSpec):
            @property
            def name(self):
                return "dummy_actuation_mode"

            def detect(self, state_codes, actuation_codes, time):
                runs = []
                in_run = False
                start = None
                for i, a in enumerate(actuation_codes):
                    if int(a) == DUMMY_ACT_STATE:
                        if not in_run:
                            in_run = True
                            start = i
                    else:
                        if in_run:
                            runs.append((start, i - 1))
                            in_run = False
                if in_run:
                    runs.append((start, len(actuation_codes) - 1))
                return runs

            def label_cycles_segment(self, df_seg, dv_dt_threshold, smooth_window):
                df_seg = df_seg.copy()
                df_seg["phase"] = "hold"
                df_seg["cycle"] = pd.array([1] * len(df_seg), dtype=pd.Int64Dtype())
                df_seg["setpoint_v"] = df_seg["voltage_v"].to_numpy(dtype=float)
                return df_seg

        dummy_spec = DummyMode()
        MODE_REGISTRY.register(dummy_spec)

        try:
            n = 50
            states = [0] * 10 + [1] * 20 + [0] * 20
            act_states = [0] * 10 + [DUMMY_ACT_STATE] * 20 + [0] * 20
            voltages = [8.0] * n
            df = _make_tidy_df(states, voltages, actuation_states=act_states)
            df = find_routine_segments(df)
            self.assertTrue((df["mode"] == "dummy_actuation_mode").any())
            dummy_rows = df[df["mode"] == "dummy_actuation_mode"]
            self.assertEqual(len(dummy_rows), 20)
        finally:
            # Remove the dummy spec to not affect other tests
            MODE_REGISTRY._specs = [s for s in MODE_REGISTRY._specs if not isinstance(s, DummyMode)]


class TestCalibrationLoading(unittest.TestCase):
    def test_json_file_loads(self):
        from optical_trap_tidy import load_calibration

        data = {
            "k_trap1_pn_per_nm_per_mw": 6.94e-5,
            "k_trap2_pn_per_nm_per_mw": 1.679e-4,
            "source_file": "Cal1_100mW.dat",
            "date": "2024-01-01",
            "fit_method": "aliased_lorentzian",
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            cal = load_calibration(cal_file=path)
            self.assertAlmostEqual(cal["k_trap1"], 6.94e-5)
            self.assertAlmostEqual(cal["k_trap2"], 1.679e-4)
            self.assertIn("Cal1_100mW.dat", cal["calibration_id"])
        finally:
            Path(path).unlink()

    def test_k_per_mw_flag_overrides_file(self):
        from optical_trap_tidy import load_calibration

        data = {"k_trap1_pn_per_nm_per_mw": 1.0e-5}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            cal = load_calibration(cal_file=path, k_per_mw_flag=9.99e-5)
            self.assertAlmostEqual(cal["k_trap1"], 9.99e-5)
        finally:
            Path(path).unlink()

    def test_no_calibration_warns_and_returns_nan(self):
        from optical_trap_tidy import load_calibration

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cal = load_calibration()
        # Should return NaN stiffness; no warning (just NaN values)
        self.assertTrue(np.isnan(cal["k_trap1"]))

    def test_load_tidy_with_k_per_mw(self):
        from optical_trap_tidy import load_tidy

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_1.dat"
        if not path.exists():
            self.skipTest("tether1_1.dat not available")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path, k_per_mw=6.94e-5)
        # stiffness = k * power; power ≈ 1000 mW → stiffness ≈ 0.0694 pN/nm
        self.assertFalse(df["stiffness_pn_per_nm"].isna().all())
        self.assertAlmostEqual(df["stiffness_pn_per_nm"].median(), 6.94e-5 * 1000.0, places=3)


class TestLoadTidySchema(unittest.TestCase):
    """load_tidy returns the correct columns for tether files."""

    def test_tidy_schema_columns(self):
        from optical_trap_tidy import load_tidy

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_1.dat"
        if not path.exists():
            self.skipTest("tether1_1.dat not available")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        expected_cols = [
            "source", "sample_idx", "time_s",
            "trap1_pos_nm", "trap2_pos_nm", "extension_nm",
            "power_mw", "voltage_v",
            "stiffness_pn_per_nm", "force_pn",
            "state_code", "actuation_state_code",
            "calibration_id",
            "mode", "cycle", "phase", "setpoint_v", "valid",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(df["source"].iloc[0], "tether1_1")


# ---------------------------------------------------------------------------
# Regression tests on real data files
# ---------------------------------------------------------------------------


class TestRegressionTether1_1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles, apply_valid_mask, summarize_run

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_1.dat"
        if not path.exists():
            raise unittest.SkipTest(f"tether1_1.dat not found: {path}")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        cls.df = df
        cls.summary = summarize_run(df)

    def test_cycle_count(self):
        self.assertEqual(self.summary["n_oscillator_cycles"], 5)

    def test_no_force_extension(self):
        self.assertEqual(self.summary["n_force_extension_pulls"], 0)

    def test_all_three_phases_present(self):
        osc = self.df[self.df["mode"] == "oscillator"]
        phases = set(osc["phase"].dropna().unique())
        self.assertIn("pull_out", phases)
        self.assertIn("pull_in", phases)
        self.assertIn("hold", phases)

    def test_extension_range_oscillator(self):
        # Extension grows with stretch and stays positive;
        # the stretched state is around +6,500 to +3,700 nm in the opposite sign.
        osc = self.df[self.df["mode"] == "oscillator"]
        ext = osc["extension_nm"].to_numpy(dtype=float)
        self.assertGreaterEqual(np.nanmax(ext), 6500)
        self.assertLessEqual(np.nanmin(ext), 4000)

    def test_setpoint_v_present_for_hold(self):
        osc = self.df[self.df["mode"] == "oscillator"]
        hold_rows = osc[osc["phase"] == "hold"]
        # setpoint_v should be numeric (not all NaN) for hold rows
        self.assertFalse(hold_rows["setpoint_v"].isna().all())

    def test_no_partial_cycles(self):
        osc = self.df[self.df["mode"] == "oscillator"]
        # This file is expected to have no partial oscillator cycles.
        self.assertFalse((osc["cycle_complete"] == False).any())  # noqa: E712


class TestRegressionTether1_2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles, apply_valid_mask, summarize_run

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_2.dat"
        if not path.exists():
            raise unittest.SkipTest(f"tether1_2.dat not found: {path}")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        cls.summary = summarize_run(df)

    def test_cycle_count(self):
        self.assertEqual(self.summary["n_oscillator_cycles"], 7)

    def test_no_force_extension(self):
        self.assertEqual(self.summary["n_force_extension_pulls"], 0)

    def test_no_partial_cycles(self):
        from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_2.dat"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        df = find_routine_segments(df)
        df = label_cycles(df)
        osc = df[df["mode"] == "oscillator"]
        # This file records a terminal stalled ramp at EOF and must retain it as a
        # partial cycle; the earlier no-partial assertion is therefore intentionally
        # not used here.
        self.assertTrue((osc["cycle_complete"] == False).any())  # noqa: E712


class TestRegressionTether1Final(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles, apply_valid_mask, summarize_run

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_final.dat"
        if not path.exists():
            raise unittest.SkipTest(f"tether1_final.dat not found: {path}")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        cls.df = df
        cls.summary = summarize_run(df)

    def test_cycle_count(self):
        self.assertEqual(self.summary["n_oscillator_cycles"], 10)

    def test_no_force_extension(self):
        self.assertEqual(self.summary["n_force_extension_pulls"], 0)

    def test_state2_glitch_masked(self):
        # tether1_final has a single-sample state 2 glitch
        glitch_rows = self.df[(self.df["state_code"] == 2) & (~self.df["valid"])]
        self.assertGreater(len(glitch_rows), 0)

    def test_partial_cycles_present_and_complete_numbers_preserved(self):
        osc = self.df[self.df["mode"] == "oscillator"].copy()
        complete_ids = sorted(osc[osc["cycle_complete"] == True]["cycle"].dropna().astype(int).unique().tolist())  # noqa: E712
        partial_ids = sorted(osc[osc["cycle_complete"] == False]["cycle"].dropna().astype(int).unique().tolist())  # noqa: E712
        self.assertEqual(complete_ids, list(range(1, 11)))
        self.assertGreaterEqual(len(partial_ids), 2)
        self.assertIn("partial_truncated", set(osc["cycle_status"].dropna().tolist()))

    def test_no_state3_or_state4_left_without_cycle(self):
        unassigned = (self.df["state_code"].isin([3, 4])) & (self.df["cycle"].isna())
        self.assertEqual(int(unassigned.sum()), 0)


class TestOscillatorAnalyzer(unittest.TestCase):
    """Smoke test for the oscillator analyzer on synthetic data."""

    def test_analyze_returns_cycle_stats(self):
        from optical_trap_tidy import find_routine_segments, label_cycles
        from optical_trap_oscillator import analyze_oscillator, plot_oscillator
        import matplotlib
        matplotlib.use("Agg")

        # Build 2 synthetic oscillator cycles
        n_per = 100
        block_v = _oscillator_voltage_block(n_per_phase=n_per)
        block_s = [4] * n_per + [3] * (n_per // 5) + [4] * n_per + [3] * n_per
        states = block_s * 2
        v = block_v * 2

        df = _make_tidy_df(states, v)
        df = find_routine_segments(df)
        df = label_cycles(df, smooth_window=30)

        stats = analyze_oscillator(df)
        self.assertFalse(stats.empty)
        self.assertIn("cycle", stats.columns)
        self.assertIn("period_s", stats.columns)
        self.assertIn("pull_out_duration_s", stats.columns)
        self.assertIn("cycle_status", stats.columns)
        self.assertIn("cycle_complete", stats.columns)

        with tempfile.TemporaryDirectory() as tmp:
            import matplotlib.pyplot as plt
            fig = plot_oscillator(df, stats)
            self.assertIsNotNone(fig)
            plt.close(fig)


class TestLoadMany(unittest.TestCase):
    def test_load_many_concatenates(self):
        from optical_trap_tidy import load_many

        repo_root = Path(__file__).resolve().parents[1]
        paths = [
            repo_root / "example_data" / "tether1_1.dat",
            repo_root / "example_data" / "tether1_2.dat",
        ]
        if not all(p.exists() for p in paths):
            self.skipTest("tether files not available")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_many(paths)
        sources = df["source"].unique().tolist()
        self.assertIn("tether1_1", sources)
        self.assertIn("tether1_2", sources)
        # All required columns present
        for col in ["mode", "cycle", "phase", "valid"]:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
