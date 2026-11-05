"""Tests for post-tidy rupture/delta analysis helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestSyntheticPostTidyAnalysis(unittest.TestCase):
    def _make_dynamic_df(self) -> pd.DataFrame:
        rows = []
        dt = 0.001
        t = 0.0
        for cyc in [1, 2]:
            # pull_out: rising force with rupture-like drop in cycle 1
            force_po = np.linspace(1.0, 25.0, 40)
            if cyc == 1:
                force_po[-5:] = np.linspace(24.0, 3.0, 5)
            ext_po = np.linspace(-5200.0, -7600.0, 40)
            for f, x in zip(force_po, ext_po):
                rows.append((t, "pull_out", cyc, f, x, 7.0))
                t += dt

            # hold
            for _ in range(20):
                rows.append((t, "hold", cyc, 8.0, -7000.0, 5.5))
                t += dt

            # pull_in
            force_pi = np.linspace(3.0, 24.0, 40)
            ext_pi = np.linspace(-7300.0, -5000.0, 40)
            for f, x in zip(force_pi, ext_pi):
                rows.append((t, "pull_in", cyc, f, x, 7.5))
                t += dt
        df = pd.DataFrame(rows, columns=["time_s", "phase", "cycle", "force_pn", "extension_nm", "voltage_v"])
        df["valid"] = True
        df["source"] = "synthetic_dynamic"
        df["mode"] = "oscillator"
        return df

    def _make_static_df(self) -> pd.DataFrame:
        time = np.arange(240, dtype=float) * 0.001
        phase = np.array(["pull_out"] * 40 + ["hold"] * 80 + ["pull_in"] * 60 + ["hold"] * 60, dtype=object)
        force = np.concatenate(
            [
                np.linspace(5.0, 14.0, 40),    # pull_out to seed analysis_cycle inference
                np.linspace(15.0, 15.0, 60),
                np.linspace(15.0, 5.0, 20),   # rupture during first hold
                np.linspace(6.0, 16.0, 60),   # pull_in used for same-force interpolation
                np.linspace(10.0, 10.0, 60),
            ]
        )
        ext = np.concatenate(
            [
                np.linspace(-6500.0, -7000.0, 40),
                np.linspace(-7000.0, -7100.0, 80),
                np.linspace(-6900.0, -6200.0, 60),
                np.linspace(-6400.0, -6500.0, 60),
            ]
        )
        df = pd.DataFrame(
            {
                "time_s": time,
                "phase": phase,
                "force_pn": force,
                "extension_nm": ext,
                "voltage_v": np.where(phase == "hold", 5.5, 7.5),
                "valid": True,
                "source": "synthetic_static",
                "mode": "puppet",
                "segment_id": np.where(
                    np.arange(len(time)) < 40,
                    1,
                    np.where(np.arange(len(time)) < 120, 2, np.where(np.arange(len(time)) < 180, 3, 4)),
                ),
            }
        )
        return df

    def test_dynamic_rupture_and_loading_rate(self):
        from optical_trap_analysis import detect_dynamic_rupture_events

        events = detect_dynamic_rupture_events(self._make_dynamic_df(), threshold_pn=8.0, n_samples=12)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(np.isfinite(events["loading_rate_pn_per_s"]).any())
        self.assertTrue((events["regime"] == "dynamic").all())

    def test_dynamic_delta_bins(self):
        from optical_trap_analysis import summarize_dynamic_delta_extension

        per_cycle, summary = summarize_dynamic_delta_extension(self._make_dynamic_df(), force_bin_target_count=10)
        self.assertFalse(per_cycle.empty)
        self.assertFalse(summary.empty)
        self.assertIn("delta_extension_nm", per_cycle.columns)
        self.assertIn("delta_extension_sd_nm", summary.columns)

    def test_static_rupture_lifetime_and_same_force(self):
        from optical_trap_analysis import (
            add_same_force_extension,
            detect_static_hold_rupture_events,
        )

        df = self._make_static_df()
        events = detect_static_hold_rupture_events(df, threshold_pn=6.0, n_samples=15)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(np.isfinite(events["lifetime_s"]).any())

        enriched = add_same_force_extension(events, df, phase="pull_in")
        self.assertIn("same_force_extension_nm", enriched.columns)
        self.assertTrue(np.isfinite(enriched["same_force_extension_nm"]).any())


class TestRealDataPostTidyAnalysis(unittest.TestCase):
    def test_real_oscillator_dynamic_delta(self):
        from optical_trap_analysis import summarize_dynamic_delta_extension
        from optical_trap_tidy import (
            apply_valid_mask,
            compute_force_reference,
            find_routine_segments,
            label_cycles,
            load_tidy,
        )

        root = Path(__file__).resolve().parents[1]
        path = root / "example_data" / "tether1_final.dat"
        df = load_tidy(path, k_per_mw=0.073)
        df = find_routine_segments(df, mode="oscillator")
        df = label_cycles(df)
        df = apply_valid_mask(df)
        df = compute_force_reference(df, method="high_hold_tail")
        osc = df[df["mode"] == "oscillator"].copy()

        per_cycle, summary = summarize_dynamic_delta_extension(osc, cycle_col="cycle", force_bin_target_count=500)
        self.assertFalse(per_cycle.empty, "Expected force-bin Δ-extension rows for real oscillator data")
        self.assertFalse(summary.empty, "Expected summarized Δ-extension bins for real oscillator data")

    def test_real_puppet_dynamic_delta(self):
        from optical_trap_analysis import summarize_dynamic_delta_extension
        from optical_trap_puppet import read_puppet

        root = Path(__file__).resolve().parents[1]
        path = root / "example_data" / "tether1_puppet.dat"
        df = read_puppet(path, stiffness=3.93e-5)

        per_cycle, summary = summarize_dynamic_delta_extension(df, force_bin_target_count=300)
        self.assertFalse(per_cycle.empty, "Expected force-bin Δ-extension rows for real puppet data")
        self.assertFalse(summary.empty, "Expected summarized Δ-extension bins for real puppet data")


if __name__ == "__main__":
    unittest.main()
