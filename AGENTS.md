# Base44 Dev Environment

## What this app is
A Python port of MATLAB optical-trap calibration code. It loads position-trace
text files plus a companion `_Voltage.dat` (trap power in mW), computes an
averaged aliased PSD, fits an aliased Lorentzian, and reports trap stiffness
(pN/nm and pN/nm/mW). The original UI is a **Tkinter desktop GUI**
(`optical_trap_calibration.py`); there is also a CLI mode (`--file`).

## Why a web view exists
The Base44 preview is a browser iframe on port 3000. A Tkinter desktop window
cannot render there, and a noVNC/VNC-over-WebSocket gateway does **not** work
through the preview proxy (it hangs the iframe). So `web_app.py` adds a thin
Flask view that **reuses `calibrate_file()` and the rest of
`optical_trap_calibration.py` unchanged** and serves an HTML page on port 3000.
The Tkinter GUI is untouched and still runnable locally with
`python optical_trap_calibration.py`.

## Running it
```
docker compose -f docker-compose.base44.yml up -d --build
```
- Web view (preview): http://localhost:3000  — pick an example file, click
  "Run calibration" to see stiffness results + the fitted PSD plot.
- The Flask dev server runs with `debug=True` / reloader, so edits to
  `web_app.py`, `templates/index.html`, and the imported `optical_trap_calibration.py`
  hot-reload without restarting the container.
- Tkinter GUI / CLI are available inside the container too, but need a display:
  `docker compose exec app python3 optical_trap_calibration.py --file example_data/example_trap_data.dat`

## Verifying it works
- `curl -sI http://localhost:3000/` → 200
- `curl -s 'http://localhost:3000/?file=Cal1_100mW.dat'` → page with
  "Trap power: 102.1 mW", a results table, and an embedded base64 PSD PNG.
- Unit tests: `docker compose exec app python3 -m unittest tests.test_optical_trap_calibration -v`
  (both tests pass).

## Dependencies
- Runtime (installed in `Dockerfile.base44`): numpy, scipy, matplotlib, flask.
  `pyinstaller` (in `requirements.txt`) is build-only and intentionally skipped.
- No external services / no secrets required.

## Files added for Base44
- `docker-compose.base44.yml`, `Dockerfile.base44`, `start.base44.sh` — runbook.
- `web_app.py`, `templates/index.html` — browser view layer (presentation only;
  all physics stays in the original modules).
