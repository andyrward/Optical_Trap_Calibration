# Optical Trap Calibration

[![Build Executables](https://github.com/andyrward/Optical_Trap_Calibration/actions/workflows/build-executables.yml/badge.svg?branch=main)](https://github.com/andyrward/Optical_Trap_Calibration/actions/workflows/build-executables.yml)

This repository contains a Python port of the MATLAB optical trap calibration code used to estimate trap stiffness from position traces using the aliased power spectrum method.

## Overview

The code keeps the original calibration workflow:

- load a position file from disk
- recover the trap power from a companion `_Voltage.dat` file
- compute the PSD from the displacement signal
- average 64 blocks of data
- fit the aliased Lorentzian spectrum using the same physical model as the MATLAB implementation
- print and display stiffness values in pN/nm and pN/nm/mW

The Python port mirrors the original MATLAB physics and defaults, including:

- `x0 = [2e-5, 100]`
- lower bounds `[1e-6, 0]`
- upper bounds `[1e-4, 1000]`
- default fit exclusion: frequencies below 20 Hz, and between 90 and 120 Hz
- aliased PSD model with the `sinc²` exposure term and harmonic sum from `j=-40` to `40`

## Module index

| Module / folder | Purpose | Main entry point | Docs |
|---|---|---|---|
| `optical_trap_calibration.py` | Tkinter GUI + CLI calibration with aliased PSD fit. | `python optical_trap_calibration.py --file ...` or `calibrate_file(...)` | This README |
| `filtpsdaliased.py` | Aliased Lorentzian PSD helper used by calibration. | `filtpsdaliased(...)` | Source docstring |
| `optical_trap_toolbox.py` | Core loading/wrangling helpers and state segmentation. | `load_normalized_trace(...)`, `segment_state_runs(...)` | [Optical Trap Toolbox](#optical-trap-toolbox-wrangling-foundation) |
| `optical_trap_state_report.py` | CLI summary of contiguous state runs. | `python optical_trap_state_report.py --file ...` | [State report utility](#state-report-utility) |
| `optical_trap_tidy.py` | Tidy per-sample builder with routine detection and analysis-oriented outputs. | `python optical_trap_tidy.py ... --mode ...` | [Analysis CLI tools](#analysis-cli-tools-optical_trap_tidypy-and-optical_trap_oscillatorpy) |
| `optical_trap_oscillator.py` | Oscillator-focused analyzer for tidy traces. | `python optical_trap_oscillator.py ...` | [Analysis CLI tools](#analysis-cli-tools-optical_trap_tidypy-and-optical_trap_oscillatorpy) |
| `optical_trap_edge_correction.py` | Edge-tracking correction used by tidy/oscillator workflows. | `correct_edge_tracking(...)` or `--edge-correction` in CLIs | [Edge-tracking correction](#edge-tracking-correction---edge-correction) |
| `optical_trap_tethers.py` | Tether ID registry, linking, and provenance helpers. | `register_or_reuse_tether(...)` and related helpers | [Tether ID and provenance flags](#tether-id-and-provenance-flags) |
| `optical_trap_puppet.py` | Puppet-mode reader that returns phase-segmented tidy frames. | `read_puppet(...)` | [Post-tidy rupture and Δ-extension analysis](#post-tidy-rupture-and-Δ-extension-analysis-optical_trap_analysispy) |
| `optical_trap_analysis.py` | Post-tidy rupture / Δ-extension analysis functions. | `detect_dynamic_rupture_events(...)` and related helpers | [Post-tidy rupture and Δ-extension analysis](#post-tidy-rupture-and-Δ-extension-analysis-optical_trap_analysispy) |
| `optical_trap_dna.py` | DNA force-extension models and distance-estimate helpers. | Module functions (no CLI) | [`docs/dna_fe_toolkit/`](docs/dna_fe_toolkit/README.md) and [`MODEL_GUIDE.md`](docs/dna_fe_toolkit/MODEL_GUIDE.md) |
| `web_app.py` + `templates/index.html` | FastAPI/Jinja web view for calibration in browser preview. | `docker compose -f docker-compose.base44.yml up -d --build` | `AGENTS.md` |
| `build_executable.py` | PyInstaller helper for local standalone builds. | `python build_executable.py` | [Local standalone executable build](#local-standalone-executable-build) |
| `Matlab_Files/` | Original MATLAB scripts and sample traces from the calibration/wrangling workflow. | MATLAB scripts (`*.m`) | [MATLAB sources and example files](#matlab-sources-and-example-files) |
| `DualTrapDataAnalysis_Eric/` | Original MATLAB dual-trap analysis codebase (legacy reference). | MATLAB scripts (`*.m`) | [MATLAB sources and example files](#matlab-sources-and-example-files) |

## Pipeline walkthrough

```text
raw trace (.dat + _Voltage.dat)
  -> calibration (`optical_trap_calibration.py`) => k (pN/nm, pN/nm/mW)
  -> tidy extraction (`optical_trap_tidy.py --mode tether-detection|oscillator`
     or `optical_trap_puppet.read_puppet`)
       + optional edge correction (`--edge-correction`)
       + optional tether IDs / provenance (`--date/--bead/--target/--registry`)
  -> post-tidy analysis (`optical_trap_analysis.py`)
       rupture force/lifetime + dynamic delta-extension summaries
  -> DNA model comparison (`optical_trap_dna.py` + `docs/dna_fe_toolkit/`)
       model deltas and distance estimate
```

Image-based particle tracking (video/microscopy to position traces) is outside this repository. Use the separate particle-tracking repository used in your workflow, then feed the resulting `.dat` traces into this repository.

## File formats

### Position data file

The expected file format is plain text with tab/space-separated values.

- Column 1: time (ignored by the calibration routine; the code re-creates the original alternating 0.710/0.720 ms sampling grid)
- Columns 2+: position data in nm

Supported formats:

- one trap, single center trace: `time, center`
- one trap, left/right edges: `time, left_edge, right_edge`
- two traps, each with a center trace or left/right pairs: additional columns are averaged in pairs
- MATLAB-style two-trap measurement: `time, left_1, left_2, right_1, right_2, const, const`

This 7-column layout is the format produced by the MATLAB code for two traps and two tracks per trap. The loader ignores the time column and averages each left/right edge pair to produce the two trap centers; `compute_averaged_psd()` reconstructs the alternating 0.710/0.720 ms grid before averaging the PSD blocks.

Example:

```
0.0000 123.4 130.2
0.0007 124.1 131.0
0.0014 125.3 129.7
```

### Voltage file

The code expects a companion file named using the original base name.

For example, if the data file is:

```
/path/to/sample.dat
```

then the calibration expects:

```
/path/to/sample_Voltage.dat
```

The first value in the voltage file is read as the trap power in mW. Real MATLAB files can store repeated alternating values such as `102.1,5.0,102.1,5.0,...`; the loader ignores the repeats and uses the first numeric value as the trap power.

Example:

```
3.14
```

## Installation

1. Clone the repository.
2. Open a terminal in the repository root.
3. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the GUI app

```bash
python optical_trap_calibration.py
```

This opens a simple Tkinter window where you can select a data file and inspect the PSD fit and stiffness results.

## Run from the command line

```bash
python optical_trap_calibration.py --file path/to/your_data.dat
```

To also display the PSD fit plot:

```bash
python optical_trap_calibration.py --file path/to/your_data.dat --plot
```

## Pre-built executables

GitHub Actions automatically builds Windows and macOS executables.

### Download options

- **Actions artifacts**: open the [Build Executables workflow](https://github.com/andyrward/Optical_Trap_Calibration/actions/workflows/build-executables.yml) and download the latest artifact from a workflow run.
- **GitHub Releases**: when a tag such as `v1.0.0` is pushed, the workflow creates a release and attaches the executables.

### Trigger a new build

- Click **Run workflow** on the Actions page for a manual build.
- Push Python changes to `main`.
- Create and push a release tag that starts with `v`.

### Platform notes

- **Windows**: Windows Defender or SmartScreen may warn the first time you run an unsigned executable.
- **macOS**: If Gatekeeper warns about an unsigned app, right-click the file and choose **Open** the first time.

## Local standalone executable build

To build locally, use the provided helper script:

```bash
python build_executable.py
```

This creates a standalone executable in the `dist/` folder using PyInstaller.

### Verifying the executable

- Run the executable and open a calibration data file.
- Confirm the trap power and stiffness values are displayed.
- For the example dataset, run:

```bash
python optical_trap_calibration.py --file example_data/example_trap_data.dat --plot
```

## Example data files

The repository includes an example dataset at:

- `example_data/example_trap_data.dat`
- `example_data/example_trap_data_Voltage.dat`

For multi-step analysis examples, use files in `example_data/` such as:

- `example_data/tether1_1.dat`
- `example_data/tether1_2.dat`
- `example_data/tether1_final.dat`
- `example_data/tether1_puppet.dat`

Note: root-level `tether1_1*.dat` files are legacy duplicates; prefer the copies in `example_data/`.

You can test the app with:

```bash
python optical_trap_calibration.py --file example_data/example_trap_data.dat --plot
```

## Optical Trap Toolbox (wrangling foundation)

This repository now includes a first-pass wrangling module:

- `optical_trap_toolbox.py`

It focuses on reusable data loading and segmentation before deeper physics analysis:

- normalize text traces into a consistent column schema:
  - `time_s, position_1_nm, position_2_nm, extension_nm, force_pn, voltage_v, stiffness_pn_per_nm, state_code, power_mw`
- support MATLAB-style calibration files and stateful six-column traces (`t, z, F, V, k, State`)
- ingest companion `_Voltage.dat` files (including repeated MATLAB alternating value streams) and align power/voltage arrays to trace length
- segment contiguous state runs with the same approach as `StateFinder.m`
- infer likely experiment phase labels from known state codes
- generate plotting summaries for:
  - state vs time
  - power/voltage vs time
  - force/extension overlays
  - segment boundaries

### State-code assumptions (v0)

The inferred labels are based on existing MATLAB code paths and are intentionally conservative:

- `21` → `tether_attempt_or_contact` (inferred from DualTrap workflow context)
- `24` → `approach`
- `27` → `measure_reference`
- `31` → `force_extension_subtether` (paired with FE logic in `TetherDataLoad.m`)
- `3` / `4` → oscillator mode regions (from `segment_regions.m`)

Unknown state codes are preserved and labeled as `unknown_state_<code>` until mapped.

## State report utility

The toolbox also includes a small report utility for inspecting contiguous state runs in a trace file:

```bash
python optical_trap_state_report.py --file Matlab_Files/tether1_part2.dat --csv state_report.csv --json state_report.json --plot state_report.png
```

This loads a MATLAB-style trace (including its companion `_Voltage.dat` if present), normalizes it to the standard schema, segments contiguous runs using the same logic as `StateFinder.m`, and prints a human-readable summary such as:

- state code
- inferred label
- confidence
- number of runs
- total samples
- total duration in seconds
- fraction of the full trace
- run boundaries (start/end index and time)

The utility supports CSV and JSON export and can optionally save a state-vs-time plot. This is intended as a lightweight inspection layer on top of the existing wrangling functions; it does not replace the calibration physics analysis.

Sample generated reports are included in the repository under `sample_reports/`:

- `sample_reports/example_trap_data_state_report.json`
- `sample_reports/example_trap_data_state_report.csv`
- `sample_reports/tether1_part2_state_report.json`
- `sample_reports/tether1_part2_state_report.csv`

These show the output format for both an example trace with no detected discrete states and a MATLAB sample trace with a single detected run.

## DNA force-extension toolkit

The repository also includes `optical_trap_dna.py`, a lightweight toolkit for single-polymer force-extension curves and looped DNA-caliper constructs. It exposes the public functions below and supports the model keywords `wlc`, `extensible_wlc`, `fjc`, and `extensible_fjc`:

- `single_polymer_extension` / `single_polymer_force`
- `full_length_curve`
- `zero_target_curve`
- `target_curve`
- `delta_extension_zero_target`
- `delta_extension_target`
- `distance_estimate`
- `compare_delta_extension`

This toolkit is documented in:

- `docs/dna_fe_toolkit/README.md` — quick-start examples and API notes
- `docs/dna_fe_toolkit/MODEL_GUIDE.md` — equations, model choices, and parameter background

Example usage:

```python
import numpy as np
from optical_trap_dna import single_polymer_extension, full_length_curve, target_curve

force = np.linspace(0.1, 20.0, 200)

ds_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}

x_ds = single_polymer_extension(force, 13250.0, 50.0, 1200.0, model="wlc", rise_per_unit_nm=0.34)
full = full_length_curve(force, 13250.0, 300.0, ds_params, ds_params)
result = target_curve(force, 13250.0, 300.0, 150.0, ds_params, ds_params, ds_params)
```

For bp or nt inputs, pass the appropriate `rise_per_unit_nm` value (for example `0.34` for dsDNA bases or `0.63` for ssDNA nucleotides).

## Packaging notes

This app was designed to be a legitimate portable replacement for the MATLAB workflow and to keep the same model-based fitting logic, including the aliased Lorentzian correction and finite exposure-time term.

## Puppet mode (`read_puppet`)

Puppet runs are voltage-programmed hold/ramp cycles. In companion
`*_params.dat`, the first non-empty row mode code `3` indicates puppet mode.
`read_acquisition_params(...)` can parse that metadata (including schedule rows),
while `read_puppet(...)` derives per-frame hold/pull phases from the actual
voltage trace.

### Minimal example (example fixture)

Files in this repository:

- `example_data/tether1_puppet.dat`
- `example_data/tether1_puppet_Voltage.dat`
- `example_data/tether1_puppet_params.dat`

```python
from pathlib import Path
from optical_trap_puppet import read_puppet

path = Path("example_data/tether1_puppet.dat")

# Option A: pass k directly (effective stiffness, pN/nm)
df = read_puppet(path, k=0.01)

# Option B: pass stiffness (pN/nm/mW); k_eff is stiffness * power_mw
# df = read_puppet(path, stiffness=3.93e-5)
```

There is currently **no puppet CLI entry point**. Use the Python API
`read_puppet(...)` directly.

### API and parameters

```python
read_puppet(
    path, k=None, stiffness=None, power=None,
    *, companion_voltage_path=None, force_ref_nm=None,
    smooth_window=500, dv_threshold=1e-4,
    rupture_threshold_nm=500.0, detect_rupture=False, source=None
)
```

- `path`: puppet `.dat` trace (MATLAB-style edge columns).
- `k` (pN/nm): effective stiffness used directly for `force_pn`.
- `stiffness` (pN/nm/mW) + `power` (mW): used as `k_eff = stiffness * power_mw`.
  If `k` is provided, it overrides `stiffness * power`.
- `companion_voltage_path`: override for the voltage/power companion file.
  If omitted, `<stem>_Voltage.dat` is used when present.
- `force_ref_nm`: optional fixed force zero (nm). Use this when you want a
  fixed force origin across all segments/runs instead of per-segment automatic
  reference updates.
- `smooth_window` (samples), `dv_threshold` (V/sample): control phase
  segmentation from the smoothed voltage derivative.
- `rupture_threshold_nm` (nm), `detect_rupture` (bool): optional rupture
  marking during `pull_out`. Off by default.
- `source`: optional run label stored in the output.

### How force and phase are defined

- Phases are inferred from `dV/dt` of the voltage schedule:
  - `hold`: `|dV/dt| <= dv_threshold`
  - `pull_out`: falling voltage (`dV/dt < -dv_threshold`)
  - `pull_in`: rising voltage (`dV/dt > dv_threshold`)
- Force uses trap 1 only:
  - `force_pn = (trap1_nm - ref_nm) * k_eff`
- If `force_ref_nm` is omitted, `ref_nm` is the mean `trap1_nm` of the
  immediately preceding hold segment, propagated over the following ramp.
  This default can shift pull-in force sign/magnitude on some runs; pass a
  fixed `force_ref_nm` when you need a stable absolute reference.

### Extension correction and output columns

`extension_nm` is preserved raw. `extension_nm_corrected` subtracts bead radii,
with each bead diameter estimated from the median inner/outer edge spacing.
Diagnostics are provided in:

- `bead1_diameter_nm`, `bead2_diameter_nm`
- `bead1_radius_nm`, `bead2_radius_nm`

Key output columns from `read_puppet(...)` include:

- provenance/indexing: `source`, `frame_idx`, `time_s`, `valid`
- geometry/signals: `trap1_nm`, `trap2_nm`, `extension_nm`,
  `extension_nm_corrected`, `voltage_v`, `power_mw`
- segmentation/force: `segment_id`, `phase`, `hold_voltage`, `k_eff`,
  `force_ref_nm`, `force_pn`
- optional rupture flag: `ruptured` (only when `detect_rupture=True`)

`read_puppet(...)` does not emit oscillator-style `cycle` or schedule `step`
columns. Downstream analysis infers cycle-like grouping as needed (for example,
`analysis_cycle` in rupture-event outputs).

### Downstream analysis (`optical_trap_analysis.py`)

Use puppet tidy frames as input to:

- `detect_dynamic_rupture_events(...)`
- `summarize_dynamic_delta_extension(...)`
- `detect_static_hold_rupture_events(...)`
- `add_same_force_extension(...)`
- `average_pull_in_trace(...)`

Rupture detection inside `read_puppet(...)` is optional (`detect_rupture=False`
by default). The main rupture/delta-extension workflow belongs in
`optical_trap_analysis.py` as explicit post-processing.

## Analysis CLI tools (`optical_trap_tidy.py` and `optical_trap_oscillator.py`)

### MATLAB sources and example files

- `Matlab_Files/` contains original MATLAB scripts plus historical sample traces.
- `DualTrapDataAnalysis_Eric/` contains the legacy dual-trap MATLAB analysis code used as a physics/workflow reference for this Python port.
- For reproducible Python examples in this repo, prefer `example_data/` files over root-level legacy duplicates.

### Post-tidy rupture and Δ-extension analysis (`optical_trap_analysis.py`)

Rupture and kinetics analysis is available as explicit post-processing on tidy
tables (oscillator or puppet), without side effects in loaders.

```python
from optical_trap_analysis import (
    detect_dynamic_rupture_events,
    detect_static_hold_rupture_events,
    summarize_dynamic_delta_extension,
    add_same_force_extension,
    average_pull_in_trace,
)
```

Typical flow:
1. Build tidy frames (`load_tidy(...)` + cycle/force labelling, or `read_puppet(...)`).
2. Dynamic regime: `detect_dynamic_rupture_events(...)` for rupture force + loading rate.
3. Dynamic Δx bins: `summarize_dynamic_delta_extension(...)` for pull_out − pull_in per force bin.
4. Static regime: `detect_static_hold_rupture_events(...)` for hold rupture lifetime/force.
5. Same-force comparison: `add_same_force_extension(...)` to interpolate extension on later pull-in at rupture force.
6. Optional pull-in stacking: `average_pull_in_trace(...)`.

### `--mode` (required for `optical_trap_tidy.py`)

`optical_trap_tidy.py` requires an explicit `--mode` flag. No default is assumed:

```bash
python optical_trap_tidy.py run.dat --mode oscillator
python optical_trap_tidy.py run.dat --mode tether-detection
```

- `oscillator` – expects states 3 (hold) and 4 (ramp).  Assigns `pull_out` /
  `pull_in` / `hold` phases from the state column; state 4 with falling voltage
  is `pull_out`, state 4 with rising voltage is `pull_in`.  If a state other
  than 3 or 4 appears inside an oscillator segment it is flagged and its time
  is printed.
- `tether-detection` – expects states 23–27 and 29.

`optical_trap_oscillator.py` is already oscillator-specific and therefore does **not** take a `--mode` flag.

### Folder input

Pass a folder instead of individual files to process all top-level `.dat` files
in that folder (no recursion).  `_Voltage.dat` and `_params.dat` files are
skipped automatically:

```bash
python optical_trap_tidy.py /path/to/data_folder/ --mode oscillator
```

### Edge-tracking correction (`--edge-correction`)

Off by default.  Applies the full edge-tracking correction described in the
issue to clean up sentinel values and systematic mis-tracking artefacts:

```bash
python optical_trap_tidy.py run.dat --mode oscillator --edge-correction
```

When enabled:
- Raw edges are preserved as `edge1_raw_nm` … `edge4_raw_nm`.
- Corrected flag per frame is added as `edge_flag_trap1`, `edge_flag_trap2`,
  `edge_flag` with values `none`, `A_width_fill`, `B_interp`, `B_width_fill`,
  `unrecoverable`.
- `unrecoverable` frames are treated as invalid by `apply_valid_mask`.
- The per-run summary reports the count and fraction of corrected frames.

### Tether ID and provenance flags

```bash
python optical_trap_tidy.py run.dat --mode oscillator \
    --date 20261008 --bead 2 \
    --target /path/to/output/ \
    --registry /path/to/tethers.csv
```

| Flag | Description |
|---|---|
| `--date YYYYMMDD` | Experiment date for tether ID generation.  Prompted if stdin is a TTY and this flag is omitted. |
| `--bead N` | Bead number.  Prompted if stdin is a TTY and omitted. |
| `--tether-id ID` | Explicit tether ID (overrides auto-generation from date + bead). |
| `--link-previous` | Link to the most recent active tether for this folder/bead. |
| `--target DIR` | Output directory for tethers.csv, provenance.json, session manifest. |
| `--registry FILE` | Path to the tether registry CSV. |

When `--target` is supplied, the following files are written:

- `<target>/session_manifest.json` – lists the processed files and their tether IDs.
- `<target>/provenance.json` – per-file provenance entry with code version, package versions, and all analysis parameters.
- For tether-detection mode: `<stem>_attempts.csv` and `<stem>_fe_curves.png`.

### Calibration file format

Pass a JSON or YAML file with `--cal-file`:

```json
{
    "k_trap1_pn_per_nm_per_mw": 7.95e-5
}
```

Or supply the value directly:

```bash
python optical_trap_tidy.py run.dat --mode oscillator --k-per-mw 7.95e-5
```

`k_trap1_pn_per_nm_per_mw` is the trap-1 stiffness in pN/nm/mW. Optional
`k_trap2_pn_per_nm_per_mw` can also be provided. Do **not** hard-code the
value 6.94e-5 from the test fixture; use the value from a recent calibration
fit (≈ 7.95e-5 based on a re-fit of the instrument data).

## License

This project is provided as-is for optical trap calibration workflows.
