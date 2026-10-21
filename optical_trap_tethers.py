#!/usr/bin/env python3
"""Tether ID registry, linking, and provenance tracking.

Public API
----------
generate_tether_id(date, bead, tether_number) -> str
    Build a tether ID string: ``YYYYMMDD_bp<bead>_t<n>``.

parse_folder_metadata(folder_path, data_root=None) -> dict
    Extract date and bead number from a ``<date> <descriptor>/bead <n>/``
    folder path.  Returns ``{date, bead, experiment_descriptor,
    folder_path_rel, metadata_source}``.

TetherRegistry(registry_path)
    Load / save the tether registry CSV.  One row per tether.

    .create_tether(date, bead, folder_path_rel, *, source_file,
                   fe_attempt, experiment_descriptor='', target='',
                   calibration_id='', processing_id='') -> str
        Register a new tether and return its ID.

    .link_previous(folder_path_rel, bead) -> str | None
        Return the most recent **active** tether ID for this folder/bead,
        or None if there is none.

    .add_history_event(tether_id, event_type, *, source_file='',
                       processing_id='', time_s=None, rupture_force_pn=None)
        Append an event to a tether's survival_history.

    .set_status(tether_id, status)
        Set a tether's status (``'active'``, ``'broken'``, ``'retired'``).

    .save()
        Write the registry to disk.

link_tether_id(df, tether_id) -> pd.DataFrame
    Assign *tether_id* to all rows in *df*.

write_session_manifest(entries, manifest_path)
    Write a YAML / JSON session manifest listing files and their tether IDs.

load_session(manifest_path, **load_tidy_kwargs) -> pd.DataFrame
    Load all files listed in a session manifest into one tidy DataFrame.

write_provenance(entry, provenance_path)
    Append a provenance manifest entry to a JSON file.

make_provenance_entry(source_file, parameters, tether_ids, *, processing_id=None) -> dict
    Build a provenance entry dict.

describe_provenance(df) -> dict
    Summarise which code versions and parameter sets appear in a merged table.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Tether ID convention: YYYYMMDD_bp<bead>_t<n>
# ---------------------------------------------------------------------------

_DATE_FOLDER_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"   # mandatory ISO date
    r"(?:\s+(?P<descriptor>.*))?$"      # optional free-text descriptor
)
_BEAD_FOLDER_RE = re.compile(r"^bead\s+(?P<bead>\d+)$", re.IGNORECASE)


def generate_tether_id(date: str, bead: int, tether_number: int) -> str:
    """Build a tether ID string.

    Parameters
    ----------
    date:
        ISO date string ``YYYY-MM-DD``.
    bead:
        Bead number (positive integer).
    tether_number:
        Tether number on this bead (starts at 1).

    Returns
    -------
    String like ``20261008_bp2_t1``.
    """
    date_compact = date.replace("-", "")
    return f"{date_compact}_bp{bead}_t{tether_number}"


def parse_folder_metadata(
    folder_path: str | Path,
    data_root: str | Path | None = None,
    *,
    date_override: str | None = None,
    bead_override: int | None = None,
) -> dict[str, Any]:
    """Extract date and bead number from a folder path.

    Expected layout::

        <data_root>/
          <YYYY-MM-DD> <experiment descriptor>/
            bead <n>/

    Parameters
    ----------
    folder_path:
        Absolute or relative path to the bead folder.
    data_root:
        Root directory used to compute ``folder_path_rel``.  If None, the
        ``folder_path`` itself is used as the relative path.
    date_override:
        Override the date; skips parsing from folder name.
    bead_override:
        Override the bead number; skips parsing from folder name.

    Returns
    -------
    Dict with keys:
    * ``date`` – ISO date string ``YYYY-MM-DD``
    * ``bead`` – bead number (int)
    * ``experiment_descriptor`` – free-text part of the date folder name
    * ``folder_path_rel`` – path relative to *data_root*
    * ``metadata_source`` – ``"path"``, ``"override"``, ``"prompt"``
      (``"prompt"`` means the caller obtained it interactively—this function
      never prompts itself)
    """
    fp = Path(folder_path).resolve()
    if data_root is not None:
        try:
            folder_path_rel = str(fp.relative_to(Path(data_root).resolve()))
        except ValueError:
            folder_path_rel = str(fp)
    else:
        folder_path_rel = str(fp)

    date: str | None = None
    bead: int | None = None
    experiment_descriptor = ""
    metadata_source = "path"

    # Walk up the path looking for the date folder and bead folder
    parts = fp.parts
    for i, part in enumerate(parts):
        m = _DATE_FOLDER_RE.match(part)
        if m and date is None:
            date = m.group("date")
            experiment_descriptor = (m.group("descriptor") or "").strip()

        m2 = _BEAD_FOLDER_RE.match(part)
        if m2 and bead is None:
            bead = int(m2.group("bead"))

    if date_override is not None:
        date = date_override
        metadata_source = "override"
    if bead_override is not None:
        bead = bead_override
        metadata_source = "override"

    if date is None or bead is None:
        missing = []
        if date is None:
            missing.append("date (YYYY-MM-DD via --date)")
        if bead is None:
            missing.append("bead number (positive integer via --bead)")
        raise ValueError(
            f"Could not parse {', '.join(missing)} from path '{folder_path}'. "
            "Please supply the missing values with --date and/or --bead."
        )

    # Validate date format
    try:
        datetime.date.fromisoformat(date)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date}': {exc}") from exc

    return {
        "date": date,
        "bead": int(bead),
        "experiment_descriptor": experiment_descriptor,
        "folder_path_rel": folder_path_rel,
        "metadata_source": metadata_source,
    }


# ---------------------------------------------------------------------------
# Tether registry
# ---------------------------------------------------------------------------

_REGISTRY_COLUMNS = [
    "tether_id",
    "date",
    "folder_path",
    "experiment_descriptor",
    "bead",
    "tether_number",
    "target",
    "source_file",
    "fe_attempt",
    "peak_force_pn",
    "plateau_force_pn",
    "survival_history",
    "status",
    "calibration_id",
    "processing_id",
    "notes",
]


class TetherRegistry:
    """Load, query, and save the tether registry.

    Parameters
    ----------
    registry_path:
        Path to the CSV file that stores the registry.  Created if it does not
        exist.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self._path = Path(registry_path)
        self._rows: list[dict[str, Any]] = []
        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with self._path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._rows = [dict(row) for row in reader]

    def _save_rows(self) -> None:
        """Write current rows to the registry CSV."""
        with self._path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_REGISTRY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def _next_tether_number(self, folder_path: str, bead: int) -> int:
        """Return ``max(existing tether_number) + 1`` for this folder/bead."""
        existing = [
            int(r["tether_number"])
            for r in self._rows
            if r.get("folder_path") == folder_path and int(r.get("bead", -1)) == bead
        ]
        return max(existing) + 1 if existing else 1

    def _get_row(self, tether_id: str) -> dict[str, Any] | None:
        for r in self._rows:
            if r.get("tether_id") == tether_id:
                return r
        return None

    def _check_id_collision(self, date: str, bead: int, folder_path: str) -> None:
        """Detect if the same tether_id would be created from a different folder."""
        date_compact = date.replace("-", "")
        # Look for any existing row with a matching date and bead but a
        # different folder_path — that would be an ID collision.
        for r in self._rows:
            tid = r.get("tether_id", "")
            prefix = f"{date_compact}_bp{bead}_"
            if tid.startswith(prefix) and r.get("folder_path") != folder_path:
                raise ValueError(
                    f"Tether ID collision detected: date '{date}' and bead {bead} "
                    f"already exist in folder '{r['folder_path']}', but you are "
                    f"trying to create a new tether in folder '{folder_path}'. "
                    "Use a different bead number, or supply a disambiguating date suffix."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_tether(
        self,
        date: str,
        bead: int,
        folder_path: str,
        *,
        source_file: str = "",
        fe_attempt: int | None = None,
        experiment_descriptor: str = "",
        target: str = "",
        calibration_id: str = "",
        processing_id: str = "",
        peak_force_pn: float | None = None,
        plateau_force_pn: float | None = None,
        notes: str = "",
    ) -> str:
        """Register a new tether and return its ID.

        Parameters
        ----------
        date:
            ISO date string ``YYYY-MM-DD``.
        bead:
            Bead number.
        folder_path:
            Path relative to the data root.
        source_file:
            File where this tether was formed.
        fe_attempt:
            The ``fe_attempt`` number within *source_file* where it was formed.

        Returns
        -------
        The new ``tether_id`` string.
        """
        self._check_id_collision(date, bead, folder_path)
        tether_number = self._next_tether_number(folder_path, bead)
        tether_id = generate_tether_id(date, bead, tether_number)
        history = json.dumps([{
            "event": "formed",
            "source_file": source_file,
            "fe_attempt": fe_attempt,
            "processing_id": processing_id,
            "time_s": None,
        }])
        row: dict[str, Any] = {
            "tether_id": tether_id,
            "date": date,
            "folder_path": folder_path,
            "experiment_descriptor": experiment_descriptor,
            "bead": bead,
            "tether_number": tether_number,
            "target": target,
            "source_file": source_file,
            "fe_attempt": fe_attempt if fe_attempt is not None else "",
            "peak_force_pn": peak_force_pn if peak_force_pn is not None else "",
            "plateau_force_pn": plateau_force_pn if plateau_force_pn is not None else "",
            "survival_history": history,
            "status": "active",
            "calibration_id": calibration_id,
            "processing_id": processing_id,
            "notes": notes,
        }
        # Ensure all required columns are present
        for col in _REGISTRY_COLUMNS:
            row.setdefault(col, "")
        self._rows.append(row)
        return tether_id

    def link_previous(self, folder_path: str, bead: int) -> str | None:
        """Return the most recent active tether ID for this folder and bead.

        Skips broken or retired tethers.

        Returns
        -------
        The most recently created active tether ID, or None if none exists.
        """
        candidates = [
            r for r in self._rows
            if r.get("folder_path") == folder_path
            and int(r.get("bead", -1)) == bead
            and r.get("status", "active") == "active"
        ]
        if not candidates:
            return None
        # Most recently created = highest tether_number
        best = max(candidates, key=lambda r: int(r.get("tether_number", 0)))
        return best["tether_id"]

    def add_history_event(
        self,
        tether_id: str,
        event_type: str,
        *,
        source_file: str = "",
        processing_id: str = "",
        time_s: float | None = None,
        rupture_force_pn: float | None = None,
    ) -> None:
        """Append an event to a tether's survival_history.

        Parameters
        ----------
        tether_id:
            The tether to update.
        event_type:
            One of ``"formed"``, ``"fe_survived"``, ``"fe_broke"``,
            ``"oscillator_run"``, ``"lost"``.
        """
        row = self._get_row(tether_id)
        if row is None:
            raise KeyError(f"Tether '{tether_id}' not found in registry.")
        try:
            history: list[dict[str, Any]] = json.loads(row.get("survival_history", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            history = []
        event: dict[str, Any] = {
            "event": event_type,
            "source_file": source_file,
            "processing_id": processing_id,
            "time_s": time_s,
        }
        if rupture_force_pn is not None:
            event["rupture_force_pn"] = rupture_force_pn
        history.append(event)
        row["survival_history"] = json.dumps(history)

    def set_status(self, tether_id: str, status: str) -> None:
        """Set a tether's status.

        Parameters
        ----------
        status:
            ``"active"``, ``"broken"``, or ``"retired"``.
        """
        row = self._get_row(tether_id)
        if row is None:
            raise KeyError(f"Tether '{tether_id}' not found in registry.")
        row["status"] = status

    def save(self) -> None:
        """Write the registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._save_rows()

    def as_dataframe(self) -> pd.DataFrame:
        """Return the registry as a DataFrame."""
        if not self._rows:
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        return pd.DataFrame(self._rows, columns=_REGISTRY_COLUMNS)


# ---------------------------------------------------------------------------
# link_tether_id
# ---------------------------------------------------------------------------


def link_tether_id(df: pd.DataFrame, tether_id: str | None) -> pd.DataFrame:
    """Assign *tether_id* to every row in *df*.

    Parameters
    ----------
    df:
        Tidy DataFrame.
    tether_id:
        The tether ID to assign.  Pass ``None`` to explicitly leave rows
        unlinked (tether_id stays NA).

    Returns
    -------
    New DataFrame with ``tether_id`` column set.
    """
    df = df.copy()
    if "tether_id" not in df.columns:
        df["tether_id"] = pd.NA
    if tether_id is not None:
        df["tether_id"] = tether_id
    return df


# ---------------------------------------------------------------------------
# Session manifest
# ---------------------------------------------------------------------------


def write_session_manifest(
    entries: list[dict[str, Any]],
    manifest_path: str | Path,
) -> None:
    """Write a session manifest listing files and their tether IDs.

    Each entry in *entries* should be a dict with at least:
    * ``source_file`` – path to the ``.dat`` file
    * ``tether_id`` – the linked tether ID (or None)

    The manifest is written as JSON.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"entries": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def load_session(
    manifest_path: str | Path,
    k_per_mw: float | None = None,
    cal_file: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load all files listed in a session manifest into one tidy DataFrame.

    Parameters
    ----------
    manifest_path:
        Path to the session manifest JSON file written by
        :func:`write_session_manifest`.
    k_per_mw:
        Trap-1 stiffness per mW forwarded to :func:`~optical_trap_tidy.load_tidy`.
    cal_file:
        Calibration file path forwarded to :func:`~optical_trap_tidy.load_tidy`.

    Returns
    -------
    Concatenated tidy DataFrame with one ``tether_id`` column.
    """
    from optical_trap_tidy import load_tidy, find_routine_segments, label_cycles, apply_valid_mask

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames: list[pd.DataFrame] = []
    for entry in manifest.get("entries", []):
        source_file = entry.get("source_file", "")
        tether_id = entry.get("tether_id")

        df = load_tidy(source_file, k_per_mw=k_per_mw, cal_file=cal_file)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        df = link_tether_id(df, tether_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


def _get_git_version() -> dict[str, str]:
    """Return the current git commit hash and dirty flag."""
    result = {"commit": "unknown", "dirty": False}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        result["commit"] = commit
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        result["dirty"] = bool(status)
    except Exception:
        pass
    return result


def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unavailable"


def make_provenance_entry(
    source_file: str | Path,
    parameters: dict[str, Any],
    tether_ids: list[str],
    *,
    processing_id: str | None = None,
) -> dict[str, Any]:
    """Build a provenance manifest entry.

    Parameters
    ----------
    source_file:
        Path to the ``.dat`` file being processed.
    parameters:
        All analysis parameters used (after defaults are applied).
    tether_ids:
        List of tether IDs assigned during this processing run.
    processing_id:
        Unique processing run ID.  A UUID4 is generated if None.

    Returns
    -------
    Provenance entry dict.
    """
    if processing_id is None:
        processing_id = str(uuid.uuid4())

    source_path = Path(source_file)
    voltage_path = source_path.parent / (source_path.stem + "_Voltage.dat")

    git = _get_git_version()

    # Package versions
    pkg_versions: dict[str, str] = {}
    for pkg in ("numpy", "scipy", "pandas", "pyarrow"):
        try:
            import importlib.metadata as _ilm
            pkg_versions[pkg] = _ilm.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    return {
        "processing_id": processing_id,
        "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_file": {
            "path": str(source_path),
            "sha256_dat": _sha256_file(source_path),
            "sha256_voltage": _sha256_file(voltage_path),
        },
        "code_version": {
            "git_commit": git["commit"],
            "dirty": git["dirty"],
        },
        "python_version": platform.python_version(),
        "package_versions": pkg_versions,
        "parameters": parameters,
        "tether_ids": tether_ids,
    }


def write_provenance(
    entry: dict[str, Any],
    provenance_path: str | Path,
) -> None:
    """Append a provenance entry to a JSON file.

    If the file already exists its contents are read, the entry is appended,
    and the file is rewritten.  The JSON structure is a list of entries.

    Parameters
    ----------
    entry:
        Provenance entry dict (as returned by :func:`make_provenance_entry`).
    provenance_path:
        Path to the provenance JSON file.
    """
    ppath = Path(provenance_path)
    ppath.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    if ppath.exists():
        try:
            entries = json.loads(ppath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            entries = []
    entries.append(entry)
    ppath.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")


def describe_provenance(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise which code versions and parameter sets appear in *df*.

    Reads the ``processing_id`` column and any ``provenance_path`` column.
    Returns a dict with unique processing IDs and any detected differences.

    Parameters
    ----------
    df:
        Tidy DataFrame with a ``processing_id`` column.

    Returns
    -------
    Dict with ``processing_ids`` (list), ``n_unique_processing_ids`` (int),
    and a ``warning`` string when multiple processing runs are detected.
    """
    if "processing_id" not in df.columns:
        return {
            "processing_ids": [],
            "n_unique_processing_ids": 0,
            "warning": "No 'processing_id' column in DataFrame.",
        }

    unique_ids = df["processing_id"].dropna().unique().tolist()
    result: dict[str, Any] = {
        "processing_ids": unique_ids,
        "n_unique_processing_ids": len(unique_ids),
    }
    if len(unique_ids) > 1:
        result["warning"] = (
            f"DataFrame contains rows from {len(unique_ids)} different processing "
            "runs. Results may have been produced with different code versions or "
            "parameters. Check provenance.json for details."
        )
        warnings.warn(result["warning"], stacklevel=2)

    return result
