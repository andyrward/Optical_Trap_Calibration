"""Tests for optical_trap_puppet.py.

Covers:
* Real-file load (example_data/tether1_puppet.dat + companion Voltage.dat).
* Column mapping matches existing toolbox convention.
* Sentinel −30.27 frames are flagged valid=False and kept.
* Phase segmentation finds holds / pull_out / pull_in segments.
* Hold segments present in the tidy output.
* Missing companion file with embedded voltage columns (synthetic fixture).
* Missing companion file AND no embedded voltage → informative error.
* Force dual-zero reference (per-segment).
* Rupture detection on synthetic data; no rupture for tether1_puppet.
* Col7 blank / all-zeros files load fine.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Path to the shared example-data directory
_EXAMPLE_DIR = Path(__file__).parent.parent / "example_data"
_PUPPET_DAT = _EXAMPLE_DIR / "tether1_puppet.dat"
_PUPPET_VOLT = _EXAMPLE_DIR / "tether1_puppet_Voltage.dat"


def _write_dat(path: Path, data: np.ndarray) -> None:
    """Write a whitespace-separated numeric table."""
    np.savetxt(str(path), data, delimiter=" ", fmt="%.6f")


def _write_volt(path: Path, power_v: np.ndarray, voltage_v: np.ndarray) -> None:
    """Write alternating power,voltage flat stream (MATLAB _Voltage.dat format)."""
    interleaved = np.empty(power_v.size + voltage_v.size)
    interleaved[0::2] = power_v
    interleaved[1::2] = voltage_v
    # Write as a single comma-separated line (matches actual file format)
    path.write_text(",".join(f"{v:.6f}" for v in interleaved), encoding="utf-8")


def _synthetic_7col(n: int = 200, include_sentinel: bool = False) -> np.ndarray:
    """Build a minimal 7-column .dat array."""
    time = np.arange(n, dtype=float) * 0.001
    # Simple values: all four edge columns + state + actuation
    col2 = np.full(n, 5000.0)
    col3 = np.full(n, 12000.0)
    col4 = np.full(n, 8000.0)
    col5 = np.full(n, 9000.0)
    state = np.zeros(n)
    actuation = np.zeros(n)
    data = np.column_stack([time, col2, col3, col4, col5, state, actuation])
    if include_sentinel:
        # Place sentinel at row 5 in col2
        data[5, 1] = -30.27
    return data


def _synthetic_voltage(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic voltage + power arrays for n samples."""
    # One hold + one ramp + one hold
    v = np.concatenate([
        np.full(60, 8.75),
        np.linspace(8.75, 5.5, 80),
        np.full(60, 5.5),
    ])[:n]
    p = np.full(n, 1000.0)
    return p, v


# ---------------------------------------------------------------------------
# Real-file tests
# ---------------------------------------------------------------------------


class TestRealFileLoad(unittest.TestCase):
    """Tests using the real example_data/tether1_puppet* fixture."""

    @classmethod
    def setUpClass(cls):
        if not _PUPPET_DAT.exists():
            raise unittest.SkipTest(f"Real fixture not found: {_PUPPET_DAT}")
        from optical_trap_puppet import read_puppet
        cls.df = read_puppet(_PUPPET_DAT)

    def test_loads_without_error(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(len(self.df), 0)

    def test_required_columns_present(self):
        required = {
            "frame_idx", "time_s", "trap1_nm", "trap2_nm",
            "extension_nm", "voltage_v", "power_mw", "valid",
            "segment_id", "phase", "hold_voltage", "force_pn",
            "ruptured", "source",
        }
        self.assertTrue(required.issubset(self.df.columns), required - set(self.df.columns))

    def test_row_count_matches_dat_file(self):
        data = np.loadtxt(str(_PUPPET_DAT))
        self.assertEqual(len(self.df), data.shape[0])

    def test_source_is_file_stem(self):
        self.assertEqual(self.df["source"].iloc[0], "tether1_puppet")


class TestColumnMapping(unittest.TestCase):
    """Numerical check: trap1, trap2, extension match MATLAB toolbox convention."""

    def test_trap_centers_match_toolbox(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        from optical_trap_toolbox import load_normalized_trace

        df = read_puppet(_PUPPET_DAT)
        trace = load_normalized_trace(_PUPPET_DAT)

        np.testing.assert_allclose(
            df["trap1_nm"].values, trace["position_1_nm"], rtol=1e-6,
            err_msg="trap1_nm should equal (col2+col4)/2"
        )
        np.testing.assert_allclose(
            df["trap2_nm"].values, trace["position_2_nm"], rtol=1e-6,
            err_msg="trap2_nm should equal (col3+col5)/2"
        )
        np.testing.assert_allclose(
            df["extension_nm"].values, trace["extension_nm"], rtol=1e-6,
            err_msg="extension_nm should equal trap2 - trap1"
        )


class TestSentinelFlagging(unittest.TestCase):
    """Frames with −30.27 must be flagged valid=False and kept."""

    def test_sentinel_count(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        # The real file is known to have at least one sentinel frame
        n_invalid = (~df["valid"]).sum()
        self.assertGreaterEqual(n_invalid, 1, "Expected at least one invalid frame")
        # Total row count must be unchanged (frames are not dropped)
        data = np.loadtxt(str(_PUPPET_DAT))
        self.assertEqual(len(df), data.shape[0])

    def test_synthetic_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = _synthetic_7col(n=50, include_sentinel=True)
            dat_path = d / "run_Puppet.dat"
            _write_dat(dat_path, data)
            p, v = _synthetic_voltage(n=50)
            _write_volt(d / "run_Puppet_Voltage.dat", p, v)
            from optical_trap_puppet import read_puppet
            df = read_puppet(dat_path)
        self.assertFalse(df["valid"].iloc[5], "Row 5 should be invalid (sentinel)")
        self.assertTrue(df["valid"].iloc[6], "Row 6 should still be valid")
        self.assertEqual(len(df), 50)


class TestPhaseSegmentation(unittest.TestCase):
    """Segmentation finds holds, pull_out, and pull_in segments."""

    def test_phases_present_in_real_file(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        phases = set(df["phase"].unique())
        self.assertIn("hold", phases)
        self.assertIn("pull_out", phases)
        self.assertIn("pull_in", phases)

    def test_hold_voltage_set_for_holds(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        holds = df[df["phase"] == "hold"]
        self.assertTrue(np.all(np.isfinite(holds["hold_voltage"])),
                        "All hold frames should have a finite hold_voltage")

    def test_hold_voltage_nan_for_ramps(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        ramps = df[df["phase"] != "hold"]
        self.assertTrue(np.all(np.isnan(ramps["hold_voltage"])),
                        "Ramp frames should have NaN hold_voltage")

    def test_hold_voltages_near_expected(self):
        """Holds should be detected near 8.75 V and 5.5 V."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        hold_levels = np.sort(np.unique(np.round(df[df["phase"] == "hold"]["hold_voltage"].dropna(), 2)))
        self.assertTrue(np.any(np.isclose(hold_levels, 8.75, atol=0.1)),
                        "Expected high-voltage hold near 8.75 V")
        self.assertTrue(np.any(np.isclose(hold_levels, 5.5, atol=0.1)),
                        "Expected low-voltage hold near 5.5 V")

    def test_hold_segments_present(self):
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        hold_segs = df[df["phase"] == "hold"]["segment_id"].unique()
        self.assertGreater(len(hold_segs), 1, "Multiple hold segments expected")

    def test_pull_directions(self):
        """pull_out segments should have falling voltage; pull_in rising."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        for phase, expected_sign in [("pull_out", -1), ("pull_in", +1)]:
            seg = df[df["phase"] == phase]
            if seg.empty:
                continue
            # Take one representative segment and check overall direction
            first_sid = seg["segment_id"].iloc[0]
            seg1 = seg[seg["segment_id"] == first_sid]
            dv = seg1["voltage_v"].iloc[-1] - seg1["voltage_v"].iloc[0]
            self.assertEqual(
                int(np.sign(dv)), expected_sign,
                f"{phase} segment voltage should move in direction {expected_sign:+d}"
            )


class TestVoltageSourceFallback(unittest.TestCase):
    """Voltage source resolution: companion → embedded columns → error."""

    def test_no_companion_with_embedded_voltage(self):
        """Newer format (>7 cols) with embedded voltage loads without error."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 100
            # Build a 9-column file: time + 4 edge + state + actuation + power + voltage
            time = np.arange(n, dtype=float) * 0.001
            edge = np.tile([5000.0, 12000.0, 8000.0, 9000.0], (n, 1))
            state = np.zeros((n, 1))
            act = np.zeros((n, 1))
            pwr = np.full((n, 1), 1000.0)
            v_col = np.concatenate([
                np.full(40, 8.75), np.linspace(8.75, 5.5, 20), np.full(40, 5.5)
            ])[:n, None]
            data9 = np.hstack([time[:, None], edge, state, act, pwr, v_col])
            dat_path = d / "run9.dat"
            np.savetxt(str(dat_path), data9, fmt="%.6f")
            # No companion file!
            df = read_puppet(dat_path)
        self.assertFalse(np.all(np.isnan(df["voltage_v"])), "Voltage should be loaded from col9")

    def test_no_voltage_source_raises(self):
        """Without any voltage source, read_puppet must raise ValueError."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = _synthetic_7col(n=50)
            dat_path = d / "no_volt.dat"
            _write_dat(dat_path, data)
            # No companion file, no embedded voltage
            with self.assertRaises(ValueError, msg="Expected ValueError when no voltage source"):
                read_puppet(dat_path)

    def test_companion_takes_precedence(self):
        """When companion file exists, it is used even if extra columns are present."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 80
            time = np.arange(n, dtype=float) * 0.001
            edge = np.tile([5000.0, 12000.0, 8000.0, 9000.0], (n, 1))
            state = np.zeros((n, 1))
            act = np.zeros((n, 1))
            pwr = np.full((n, 1), 1000.0)
            # Embedded voltage = constant 3.0 V (clearly distinct)
            embedded_v = np.full((n, 1), 3.0)
            data9 = np.hstack([time[:, None], edge, state, act, pwr, embedded_v])
            dat_path = d / "run9b.dat"
            np.savetxt(str(dat_path), data9, fmt="%.6f")
            # Companion file has proper puppet voltages (8.75/5.5)
            companion_v = np.concatenate([np.full(40, 8.75), np.full(40, 5.5)])
            companion_p = np.full(n, 1007.0)
            _write_volt(d / "run9b_Voltage.dat", companion_p, companion_v)
            df = read_puppet(dat_path)
        # Should use companion (8.75/5.5), not embedded (3.0)
        self.assertAlmostEqual(df["voltage_v"].max(), 8.75, places=2)


class TestForce(unittest.TestCase):
    """Force computation and dual-zero references."""

    def test_k_direct_overrides_stiffness_power(self):
        """Passing k directly should set k_eff uniformly."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        k_val = 0.05
        df = read_puppet(_PUPPET_DAT, k=k_val)
        np.testing.assert_allclose(df["k_eff"].values, k_val,
                                   err_msg="k_eff should equal k when k is supplied directly")

    def test_stiffness_times_power(self):
        """k_eff = stiffness × power when k is not supplied."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        stiffness = 3.93e-5
        df = read_puppet(_PUPPET_DAT, stiffness=stiffness)
        expected_k = stiffness * df["power_mw"].values
        np.testing.assert_allclose(df["k_eff"].values, expected_k, rtol=1e-6)

    def test_force_pn_nan_without_stiffness(self):
        """force_pn is NaN when no stiffness is provided."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT)
        self.assertTrue(np.all(np.isnan(df["force_pn"])),
                        "force_pn must be NaN when no stiffness is supplied")

    def test_force_ref_changes_between_segments(self):
        """The force reference should differ between pull_out and pull_in phases."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT, k=0.01)
        po = df[df["phase"] == "pull_out"]["force_ref_nm"].dropna()
        pi = df[df["phase"] == "pull_in"]["force_ref_nm"].dropna()
        if po.empty or pi.empty:
            self.skipTest("Not enough segments to compare refs")
        # Pull-out ref (from high-voltage hold) should differ from pull-in ref
        # (from low-voltage hold) because the bead position changes
        self.assertFalse(
            np.isclose(po.mean(), pi.mean(), rtol=0.01),
            "Pull-out and pull-in force references should generally differ"
        )

    def test_force_ref_override_used(self):
        """A user-supplied force_ref_nm should override the automatic one."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        ref_nm = 123.456
        df = read_puppet(_PUPPET_DAT, k=0.01, force_ref_nm=ref_nm)
        self.assertTrue(np.allclose(df["force_ref_nm"].to_numpy(), ref_nm),
                        "force_ref_nm should be fixed when supplied")
        self.assertTrue(np.allclose(df["force_pn"].to_numpy(), (df["trap1_nm"] - ref_nm) * 0.01),
                        "force_pn should use the caller-supplied force reference")

    def test_bead_radius_correction_and_diagnostics(self):
        """Raw extension is preserved, corrected extension subtracts bead radii."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 50
            time = np.arange(n, dtype=float) * 0.001
            edge = np.tile([5000.0, 12000.0, 8000.0, 9000.0], (n, 1))
            state = np.zeros((n, 1))
            act = np.zeros((n, 1))
            pwr = np.full((n, 1), 1000.0)
            v = np.full((n, 1), 8.75)
            data = np.hstack([time[:, None], edge, state, act, pwr, v])
            dat_path = d / "radius_check.dat"
            np.savetxt(str(dat_path), data, fmt="%.6f")
            p, vtrace = _synthetic_voltage(n=n)
            _write_volt(d / "radius_check_Voltage.dat", p, vtrace)
            df = read_puppet(dat_path, k=0.01)

        raw_ext = df["extension_nm"].iloc[0]
        r1 = df["bead1_radius_nm"].iloc[0]
        r2 = df["bead2_radius_nm"].iloc[0]
        expected = raw_ext - r1 - r2
        self.assertIn("extension_nm_corrected", df.columns)
        self.assertIn("bead1_diameter_nm", df.columns)
        self.assertIn("bead2_diameter_nm", df.columns)
        self.assertAlmostEqual(df["extension_nm_corrected"].iloc[0], expected, places=6)
        self.assertGreater(df["bead1_diameter_nm"].iloc[0], 0.0)
        self.assertGreater(df["bead2_diameter_nm"].iloc[0], 0.0)


class TestRupture(unittest.TestCase):
    """Rupture detection."""

    def _make_rupture_dat(self, tmpdir: Path) -> Path:
        """Build a synthetic .dat with a simulated rupture in the pull_out segment."""
        n_hold1 = 60
        n_ramp = 120
        n_hold2 = 60
        n = n_hold1 + n_ramp + n_hold2   # 240
        time = np.arange(n, dtype=float) * 0.001
        # trap1: hold at 5000, then pull_out with a rupture (drop of 1500 nm near end)
        trap1 = np.concatenate([
            np.full(n_hold1, 5000.0),
            np.linspace(5000.0, 4200.0, n_ramp),   # pull_out, trap1 decreasing
            np.full(n_hold2, 4200.0 - 1500.0),      # after rupture: large drop
        ])
        trap2 = np.full(n, 12000.0)
        state = np.zeros(n)
        act = np.zeros(n)
        dat_path = tmpdir / "rupture.dat"
        data = np.column_stack([time, trap1, trap2, trap1, trap2, state, act])
        _write_dat(dat_path, data)
        # Voltage: hold (high) → pull_out (falling) → hold (low)
        v = np.concatenate([
            np.full(n_hold1, 8.75),
            np.linspace(8.75, 5.5, n_ramp),
            np.full(n_hold2, 5.5),
        ])
        p = np.full(n, 1000.0)
        _write_volt(tmpdir / "rupture_Voltage.dat", p, v)
        return dat_path

    def test_rupture_detected(self):
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            dat_path = self._make_rupture_dat(Path(tmp))
            df = read_puppet(dat_path, rupture_threshold_nm=1000.0)
        self.assertTrue(df["ruptured"].any(), "Expected a rupture to be detected")

    def test_no_rupture_for_real_file(self):
        """tether1_puppet.dat should not have a rupture (extension returns cleanly)."""
        if not _PUPPET_DAT.exists():
            self.skipTest("Real fixture not found")
        from optical_trap_puppet import read_puppet
        df = read_puppet(_PUPPET_DAT, rupture_threshold_nm=500.0)
        self.assertFalse(df["ruptured"].any(), "tether1_puppet.dat should not have a rupture")

    def test_rupture_keeps_partial_data(self):
        """Data after a rupture should remain in the table."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            dat_path = self._make_rupture_dat(Path(tmp))
            df = read_puppet(dat_path, rupture_threshold_nm=1000.0)
        # The 60-sample low-voltage hold after the rupture should still be present
        ruptured_rows = df[df["ruptured"]]
        self.assertGreater(len(ruptured_rows), 0, "Post-rupture rows must be kept")


class TestCol7Handling(unittest.TestCase):
    """Files with col7 blank or all zeros must load fine."""

    def test_all_zero_col7(self):
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = _synthetic_7col(n=50)
            # col7 (actuation state, index 6) all zeros already in _synthetic_7col
            dat_path = d / "zeros7.dat"
            _write_dat(dat_path, data)
            p, v = _synthetic_voltage(n=50)
            _write_volt(d / "zeros7_Voltage.dat", p, v)
            df = read_puppet(dat_path)
        self.assertEqual(len(df), 50)

    def test_6col_no_col7(self):
        """If only 6 real data columns are present (no actuation state), load fine."""
        from optical_trap_puppet import read_puppet
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 50
            time = np.arange(n, dtype=float) * 0.001
            edge = np.tile([5000.0, 12000.0, 8000.0, 9000.0], (n, 1))
            state = np.zeros((n, 1))
            data6 = np.hstack([time[:, None], edge, state])
            dat_path = d / "no_act.dat"
            np.savetxt(str(dat_path), data6, fmt="%.6f")
            p, v = _synthetic_voltage(n=n)
            _write_volt(d / "no_act_Voltage.dat", p, v)
            df = read_puppet(dat_path)
        self.assertEqual(len(df), n)


if __name__ == "__main__":
    unittest.main()
