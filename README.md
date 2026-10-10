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
- `24` → `reference_pull` (from `StateFinder(24, ...)` usage in `TetherDataLoad.m`)
- `27` → `force_extension` (from FE extraction in `TetherDataLoad.m`)
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

## Packaging notes

This app was designed to be a legitimate portable replacement for the MATLAB workflow and to keep the same model-based fitting logic, including the aliased Lorentzian correction and finite exposure-time term.

## License

This project is provided as-is for optical trap calibration workflows.
