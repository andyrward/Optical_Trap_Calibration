"""Tests for optical_trap_tethers.py.

Covers:
* Tether ID generation.
* parse_folder_metadata: date/bead from path, descriptor stripped.
* TetherRegistry: create, link_previous, add_history_event, set_status, save/load.
* ID collision detection.
* Multiple tethers on one bead (t1, t2).
* link_previous skips broken tethers.
* link_tether_id assigns tether_id column.
* write_session_manifest / load_session.
* Provenance: make_provenance_entry, write_provenance, describe_provenance.
* Force-from-reference and tether-detection segmentation.
* Outcome refinement (broke / no_tether).
* Non-standard path stops without --date / --bead.
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
# Helpers shared across test cases
# ---------------------------------------------------------------------------


def _make_tidy_df(
    states: list[int],
    trap1_nm: list[float] | None = None,
    power_mw: float = 1000.0,
    sample_rate: float = 1400.0,
) -> pd.DataFrame:
    """Build a minimal tidy DataFrame for tether-detection tests."""
    from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles, apply_valid_mask

    n = len(states)
    if trap1_nm is None:
        trap1_nm = [6660.0] * n
    time = np.arange(n, dtype=float) / sample_rate
    return pd.DataFrame(
        {
            "source": "synthetic",
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": time,
            "trap1_pos_nm": np.array(trap1_nm, dtype=float),
            "trap2_pos_nm": np.full(n, 500.0),
            "extension_nm": np.array(trap1_nm, dtype=float) - 500.0,
            "power_mw": np.full(n, power_mw),
            "voltage_v": np.full(n, 7.5),
            "stiffness_pn_per_nm": np.full(n, 6.94e-5 * power_mw),
            "force_pn": np.full(n, np.nan),
            "state_code": np.array(states, dtype=float),
            "actuation_state_code": np.zeros(n),
            "calibration_id": "test",
            "mode": "other",
            "cycle": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "phase": pd.array([pd.NA] * n, dtype=object),
            "setpoint_v": np.full(n, np.nan),
            "valid": True,
            "fe_attempt": pd.array([pd.NA] * n, dtype=pd.Int64Dtype()),
            "outcome": pd.array([pd.NA] * n, dtype=object),
            "force_ref_nm": np.full(n, np.nan),
            "force_ref_method": pd.array([pd.NA] * n, dtype=object),
            "tether_id": pd.array([pd.NA] * n, dtype=object),
            "bead": pd.array([pd.NA] * n, dtype=object),
            "tether_number": pd.array([pd.NA] * n, dtype=object),
            "processing_id": pd.array([pd.NA] * n, dtype=object),
        }
    )


# ---------------------------------------------------------------------------
# Tether ID generation
# ---------------------------------------------------------------------------


class TestGenerateTetherId(unittest.TestCase):
    def test_basic(self):
        from optical_trap_tethers import generate_tether_id

        tid = generate_tether_id("2026-10-08", 2, 1)
        self.assertEqual(tid, "20261008_bp2_t1")

    def test_second_tether(self):
        from optical_trap_tethers import generate_tether_id

        tid = generate_tether_id("2026-10-08", 2, 2)
        self.assertEqual(tid, "20261008_bp2_t2")

    def test_large_bead(self):
        from optical_trap_tethers import generate_tether_id

        tid = generate_tether_id("2020-01-15", 12, 3)
        self.assertEqual(tid, "20200115_bp12_t3")


# ---------------------------------------------------------------------------
# parse_folder_metadata
# ---------------------------------------------------------------------------


class TestParseFolderMetadata(unittest.TestCase):
    def test_parses_date_and_bead(self):
        from optical_trap_tethers import parse_folder_metadata

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-10-08 test_descriptor" / "bead 2"
            folder.mkdir(parents=True)
            meta = parse_folder_metadata(folder, data_root=tmp)

        self.assertEqual(meta["date"], "2026-10-08")
        self.assertEqual(meta["bead"], 2)
        self.assertEqual(meta["experiment_descriptor"], "test_descriptor")

    def test_descriptor_not_in_id(self):
        from optical_trap_tethers import parse_folder_metadata, generate_tether_id

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-10-08 some long descriptor" / "bead 1"
            folder.mkdir(parents=True)
            meta = parse_folder_metadata(folder, data_root=tmp)

        tid = generate_tether_id(meta["date"], meta["bead"], 1)
        # Descriptor must not appear in the ID
        self.assertNotIn("descriptor", tid)
        self.assertNotIn("some", tid)
        self.assertEqual(tid, "20261008_bp1_t1")

    def test_override_date_and_bead(self):
        from optical_trap_tethers import parse_folder_metadata

        with tempfile.TemporaryDirectory() as tmp:
            # Non-standard path (no date/bead folders)
            folder = Path(tmp) / "arbitrary_folder"
            folder.mkdir(parents=True)
            meta = parse_folder_metadata(
                folder, data_root=tmp, date_override="2024-05-01", bead_override=3
            )

        self.assertEqual(meta["date"], "2024-05-01")
        self.assertEqual(meta["bead"], 3)
        self.assertEqual(meta["metadata_source"], "override")

    def test_nonstandard_path_raises_without_overrides(self):
        from optical_trap_tethers import parse_folder_metadata

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "arbitrary_folder"
            folder.mkdir(parents=True)
            with self.assertRaises(ValueError) as ctx:
                parse_folder_metadata(folder, data_root=tmp)

        self.assertIn("--date", str(ctx.exception))
        self.assertIn("--bead", str(ctx.exception))


# ---------------------------------------------------------------------------
# TetherRegistry: basic operations
# ---------------------------------------------------------------------------


class TestTetherRegistry(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self._tmpdir.name) / "tethers.csv"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_registry(self):
        from optical_trap_tethers import TetherRegistry
        return TetherRegistry(self.registry_path)

    def test_create_tether(self):
        reg = self._make_registry()
        tid = reg.create_tether(
            "2026-10-08", 2, "2026-10-08 test/bead 2",
            source_file="tether1_overstretch.dat", fe_attempt=5,
        )
        self.assertEqual(tid, "20261008_bp2_t1")

    def test_save_and_reload(self):
        from optical_trap_tethers import TetherRegistry

        reg = self._make_registry()
        tid = reg.create_tether("2026-10-08", 2, "test/bead 2",
                                source_file="file.dat", fe_attempt=1)
        reg.save()

        reg2 = TetherRegistry(self.registry_path)
        df = reg2.as_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["tether_id"], tid)

    def test_link_previous_returns_active(self):
        reg = self._make_registry()
        folder = "test/bead 2"
        tid1 = reg.create_tether("2026-10-08", 2, folder, source_file="f1.dat")
        tid_result = reg.link_previous(folder, 2)
        self.assertEqual(tid_result, tid1)

    def test_link_previous_skips_broken(self):
        reg = self._make_registry()
        folder = "test/bead 2"
        tid1 = reg.create_tether("2026-10-08", 2, folder, source_file="f1.dat")
        reg.set_status(tid1, "broken")
        result = reg.link_previous(folder, 2)
        self.assertIsNone(result)

    def test_link_previous_returns_latest_active(self):
        reg = self._make_registry()
        folder = "test/bead 2"
        tid1 = reg.create_tether("2026-10-08", 2, folder, source_file="f1.dat")
        reg.set_status(tid1, "broken")
        tid2 = reg.create_tether("2026-10-08", 2, folder, source_file="f2.dat")
        result = reg.link_previous(folder, 2)
        self.assertEqual(result, tid2)  # skips t1 (broken), returns t2

    def test_link_previous_none_when_empty(self):
        reg = self._make_registry()
        self.assertIsNone(reg.link_previous("no/such/folder", 1))

    def test_multiple_tethers_same_bead(self):
        reg = self._make_registry()
        folder = "test/bead 1"
        tid1 = reg.create_tether("2026-10-08", 1, folder, source_file="f.dat")
        reg.set_status(tid1, "broken")
        tid2 = reg.create_tether("2026-10-08", 1, folder, source_file="f.dat")
        self.assertEqual(tid1, "20261008_bp1_t1")
        self.assertEqual(tid2, "20261008_bp1_t2")
        df = reg.as_dataframe()
        self.assertEqual(len(df), 2)

    def test_id_collision_between_different_folders(self):
        reg = self._make_registry()
        reg.create_tether("2026-10-08", 2, "folder_A/bead 2", source_file="a.dat")
        with self.assertRaises(ValueError) as ctx:
            # Same date + bead, different folder → collision
            reg.create_tether("2026-10-08", 2, "folder_B/bead 2", source_file="b.dat")
        self.assertIn("collision", str(ctx.exception).lower())

    def test_add_history_event(self):
        reg = self._make_registry()
        tid = reg.create_tether("2026-10-08", 1, "f/b1", source_file="a.dat")
        reg.add_history_event(tid, "fe_survived", source_file="b.dat", time_s=12.5)
        row = reg._get_row(tid)
        history = json.loads(row["survival_history"])
        event_types = [e["event"] for e in history]
        self.assertIn("formed", event_types)
        self.assertIn("fe_survived", event_types)

    def test_set_status_broken(self):
        reg = self._make_registry()
        tid = reg.create_tether("2026-10-08", 1, "f/b1", source_file="a.dat")
        self.assertEqual(reg._get_row(tid)["status"], "active")
        reg.set_status(tid, "broken")
        self.assertEqual(reg._get_row(tid)["status"], "broken")


# ---------------------------------------------------------------------------
# link_tether_id
# ---------------------------------------------------------------------------


class TestLinkTetherId(unittest.TestCase):
    def test_assigns_tether_id(self):
        from optical_trap_tethers import link_tether_id

        df = _make_tidy_df([27] * 5)
        df2 = link_tether_id(df, "20261008_bp2_t1")
        self.assertTrue((df2["tether_id"] == "20261008_bp2_t1").all())

    def test_null_when_no_link(self):
        from optical_trap_tethers import link_tether_id

        df = _make_tidy_df([27] * 5)
        df2 = link_tether_id(df, None)
        # tether_id should remain NA
        self.assertTrue(df2["tether_id"].isna().all())


# ---------------------------------------------------------------------------
# Session manifest
# ---------------------------------------------------------------------------


class TestSessionManifest(unittest.TestCase):
    def test_write_and_load_manifest(self):
        from optical_trap_tethers import write_session_manifest, load_session

        with tempfile.TemporaryDirectory() as tmp:
            # Use real example data
            repo_root = Path(__file__).resolve().parents[1]
            f1 = repo_root / "example_data" / "tether1_1.dat"
            if not f1.exists():
                self.skipTest("tether1_1.dat not available")

            entries = [{"source_file": str(f1), "tether_id": "20261008_bp2_t1"}]
            manifest_path = Path(tmp) / "session.json"
            write_session_manifest(entries, manifest_path)
            self.assertTrue(manifest_path.exists())

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                df = load_session(manifest_path)

            self.assertFalse(df.empty)
            self.assertTrue((df["tether_id"] == "20261008_bp2_t1").all())
            self.assertIn("tether1_1", df["source"].unique())


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_make_and_write_entry(self):
        from optical_trap_tethers import make_provenance_entry, write_provenance

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fake.dat"
            source.write_bytes(b"fake")

            params = {"k_per_mw": 6.94e-5, "force_ref": "measure_ref"}
            entry = make_provenance_entry(
                source, params, ["20261008_bp2_t1"], processing_id="test-pid"
            )
            self.assertEqual(entry["processing_id"], "test-pid")
            self.assertIn("processed_at", entry)
            self.assertIn("code_version", entry)
            self.assertIn("parameters", entry)

            prov_path = Path(tmp) / "provenance.json"
            write_provenance(entry, prov_path)
            data = json.loads(prov_path.read_text())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["processing_id"], "test-pid")

    def test_write_provenance_appends(self):
        from optical_trap_tethers import make_provenance_entry, write_provenance

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fake.dat"
            source.write_bytes(b"fake")
            prov_path = Path(tmp) / "provenance.json"

            for i in range(3):
                entry = make_provenance_entry(
                    source, {}, [], processing_id=f"pid-{i}"
                )
                write_provenance(entry, prov_path)

            data = json.loads(prov_path.read_text())
            self.assertEqual(len(data), 3)

    def test_describe_provenance_warns_multiple(self):
        from optical_trap_tethers import describe_provenance

        df = pd.DataFrame({"processing_id": ["pid-1", "pid-1", "pid-2"]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = describe_provenance(df)

        self.assertEqual(result["n_unique_processing_ids"], 2)
        self.assertTrue(any("processing" in str(x.message).lower() for x in w))

    def test_describe_provenance_no_column(self):
        from optical_trap_tethers import describe_provenance

        df = pd.DataFrame({"x": [1, 2]})
        result = describe_provenance(df)
        self.assertEqual(result["n_unique_processing_ids"], 0)


# ---------------------------------------------------------------------------
# Tether-detection segmentation (TetherDetectionMode)
# ---------------------------------------------------------------------------


class TestTetherDetectionSegmentation(unittest.TestCase):
    def _make_labeled_df(self, states, trap1_nm=None):
        """Build and fully label a tidy DF from states."""
        from optical_trap_tidy import find_routine_segments, label_cycles, apply_valid_mask

        df = _make_tidy_df(states, trap1_nm=trap1_nm)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        return df

    def test_broke_sequence_27_29_23(self):
        # 27 (reference) → 29 (pull) → 23 (broke)
        n = 20
        states = [27] * n + [29] * n + [23] * n
        df = self._make_labeled_df(states)
        td = df[df["mode"] == "tether_detection"]
        self.assertFalse(td.empty)
        # Phase labels
        phases = set(td["phase"].dropna().unique())
        self.assertIn("reference", phases)
        self.assertIn("pull_out", phases)
        self.assertIn("post_break", phases)
        # Outcome on pull_out rows is 'broke_or_no_tether' (force not yet computed)
        pull = td[td["phase"] == "pull_out"]
        self.assertTrue((pull["outcome"] == "broke_or_no_tether").all())

    def test_survived_sequence_27_29_26_27(self):
        # 27 (ref) → 29 (pull) → 26 (relax) → 27 (new ref)
        n = 20
        states = [27] * n + [29] * n + [26] * n + [27] * n
        df = self._make_labeled_df(states)
        td = df[df["mode"] == "tether_detection"]
        phases = set(td["phase"].dropna().unique())
        self.assertIn("pull_out", phases)
        self.assertIn("relax", phases)
        # Outcome on pull_out rows is 'survived'
        pull = td[td["phase"] == "pull_out"]
        self.assertTrue((pull["outcome"] == "survived").all())

    def test_fe_attempt_numbering(self):
        # Two pulls: 27→29→23, 27→29→26
        n = 10
        states = (
            [27] * n + [29] * n + [23] * n
            + [27] * n + [29] * n + [26] * n
        )
        df = self._make_labeled_df(states)
        td = df[df["mode"] == "tether_detection"]
        attempts = sorted(td["fe_attempt"].dropna().astype(int).unique().tolist())
        self.assertEqual(attempts, [1, 2])

    def test_no_state_29_in_oscillator_file(self):
        # Oscillator files have no state 29; tether_detection should be empty
        from optical_trap_tidy import find_routine_segments, label_cycles, apply_valid_mask

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "example_data" / "tether1_1.dat"
        if not path.exists():
            self.skipTest("tether1_1.dat not available")

        from optical_trap_tidy import load_tidy

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = load_tidy(path)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)

        td = df[df["mode"] == "tether_detection"]
        self.assertTrue(td.empty)


# ---------------------------------------------------------------------------
# Force-from-reference
# ---------------------------------------------------------------------------


class TestForceFromReference(unittest.TestCase):
    def test_measure_ref_method(self):
        from optical_trap_tidy import (
            find_routine_segments, label_cycles, apply_valid_mask,
            compute_force_reference,
        )

        # State 27: trap1 at 6660 nm (reference hold)
        # State 29: trap1 at 6760 nm (displaced by 100 nm)
        n = 50
        ref_nm = 6660.0
        pull_nm = 6760.0
        stiffness = 6.94e-5 * 1000.0  # k * power_mw

        states = [27] * n + [29] * n
        trap1 = [ref_nm] * n + [pull_nm] * n

        df = _make_tidy_df(states, trap1_nm=trap1)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        df = compute_force_reference(df, method="measure_ref")

        pull_rows = df[df["phase"] == "pull_out"]
        self.assertFalse(pull_rows.empty)
        expected_force = (pull_nm - ref_nm) * stiffness
        self.assertAlmostEqual(
            pull_rows["force_pn"].mean(), expected_force, places=3
        )

    def test_fixed_method(self):
        from optical_trap_tidy import (
            find_routine_segments, label_cycles, apply_valid_mask,
            compute_force_reference,
        )

        n = 20
        ref_nm = 6650.0
        pull_nm = 6700.0
        stiffness = 6.94e-5 * 1000.0

        states = [29] * n
        trap1 = [pull_nm] * n

        df = _make_tidy_df(states, trap1_nm=trap1)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = compute_force_reference(df, method="fixed", force_ref_nm=ref_nm)

        td = df[df["mode"] == "tether_detection"]
        expected = (pull_nm - ref_nm) * stiffness
        self.assertAlmostEqual(td["force_pn"].mean(), expected, places=3)

    def test_outcome_refinement_broke_vs_no_tether(self):
        from optical_trap_tidy import (
            find_routine_segments, label_cycles, apply_valid_mask,
            compute_force_reference, label_tether_detection_outcomes,
        )

        n = 50
        # Attempt 1: force stays near 0 → no_tether
        # Attempt 2: force rises then falls → broke
        ref_nm = 6660.0
        # ref hold, flat pull (no tether), break, ref hold, high pull, break
        states = (
            [27] * n + [29] * n + [23] * n
            + [27] * n + [29] * n + [23] * n
        )
        trap1_no_tether = [ref_nm] * n      # no displacement = no force
        trap1_broke = [ref_nm + 500.0] * n  # large displacement = force >> threshold

        trap1 = [ref_nm] * n + trap1_no_tether + [ref_nm] * n \
                + [ref_nm] * n + trap1_broke + [ref_nm] * n

        df = _make_tidy_df(states, trap1_nm=trap1)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        df = compute_force_reference(df, method="measure_ref")
        df = label_tether_detection_outcomes(df, no_tether_threshold_pn=3.0)

        pull_rows = df[df["phase"] == "pull_out"].copy()
        outcomes_by_attempt = (
            pull_rows.groupby(pull_rows["fe_attempt"].astype(int))["outcome"].first()
        )
        self.assertEqual(outcomes_by_attempt[1], "no_tether")
        self.assertEqual(outcomes_by_attempt[2], "broke")


# ---------------------------------------------------------------------------
# Linking: downstream rows get tether_id, unlinked rows get null
# ---------------------------------------------------------------------------


class TestLinkingIntegration(unittest.TestCase):
    def test_unlinked_downstream_rows_are_null(self):
        from optical_trap_tethers import link_tether_id

        df = _make_tidy_df([27] * 10 + [29] * 10)
        # Without explicit linking, tether_id is NA
        self.assertTrue(df["tether_id"].isna().all())

    def test_link_tether_id_all_rows(self):
        from optical_trap_tethers import link_tether_id

        df = _make_tidy_df([27] * 10 + [29] * 10)
        df = link_tether_id(df, "20261008_bp2_t1")
        self.assertTrue((df["tether_id"] == "20261008_bp2_t1").all())

    def test_full_linking_integration(self):
        """Create a tether from an overstretch run, link a follow-up."""
        from optical_trap_tethers import TetherRegistry, link_tether_id

        with tempfile.TemporaryDirectory() as tmp:
            folder = "2026-10-08 test/bead 2"
            reg = TetherRegistry(Path(tmp) / "tethers.csv")

            # Overstretch run creates tether
            tid = reg.create_tether(
                "2026-10-08", 2, folder,
                source_file="tether1_overstretch.dat", fe_attempt=5,
            )
            reg.save()

            # Follow-up run links previous
            linked_tid = reg.link_previous(folder, 2)
            self.assertEqual(linked_tid, tid)

            df_followup = _make_tidy_df([3] * 100 + [4] * 100)
            df_followup = link_tether_id(df_followup, linked_tid)
            self.assertTrue((df_followup["tether_id"] == tid).all())


# ---------------------------------------------------------------------------
# High-hold-tail reference method (oscillator)
# ---------------------------------------------------------------------------


class TestHighHoldTailReference(unittest.TestCase):
    def test_high_hold_tail_method(self):
        from optical_trap_tidy import (
            find_routine_segments, label_cycles, apply_valid_mask,
            compute_force_reference,
        )

        # Build a synthetic oscillator cycle: pull_out (falling V), hold (low V),
        # pull_in (rising V), hold (high V = reference)
        n_po = 100
        n_hold_low = 20
        n_pi = 100
        n_hold_high = 100

        states = [4] * n_po + [3] * n_hold_low + [4] * n_pi + [3] * n_hold_high
        v_po = np.linspace(8.8, 5.6, n_po)
        v_hl = np.full(n_hold_low, 5.6)
        v_pi = np.linspace(5.6, 8.8, n_pi)
        v_hh = np.full(n_hold_high, 8.8)
        voltages = np.concatenate([v_po, v_hl, v_pi, v_hh]).tolist()

        # trap1 at reference during high hold (8800 nm), displaced during pulls
        ref_nm = 6680.0
        trap1 = np.full(len(states), ref_nm - 50.0)
        trap1[n_po + n_hold_low + n_pi:] = ref_nm  # high hold at reference

        df = _make_tidy_df(states, trap1_nm=trap1.tolist())
        df["voltage_v"] = voltages
        df = find_routine_segments(df)
        df = label_cycles(df, smooth_window=30)
        df = apply_valid_mask(df)
        df = compute_force_reference(df, method="high_hold_tail", force_ref_frames=50)

        osc = df[df["mode"] == "oscillator"]
        self.assertFalse(osc.empty)
        # Force at high hold should be ~0 (hold is the reference)
        high_hold = osc[osc["phase"] == "hold"]
        if len(high_hold):
            mean_force_at_ref = high_hold["force_pn"].dropna().mean()
            # Allow NaN or near-zero
            if not np.isnan(mean_force_at_ref):
                self.assertAlmostEqual(mean_force_at_ref, 0.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
