# DNA force-extension toolkit

This folder documents the DNA force-extension toolkit implemented in `optical_trap_dna.py`.

The toolkit is designed for rapid comparison of DNA force-extension curves and looped constructs, and it exposes the public functions listed in the repo README:

- `single_polymer_extension` / `single_polymer_force`
- `full_length_curve`
- `zero_target_curve`
- `target_curve`
- `delta_extension_zero_target`
- `delta_extension_target`
- `distance_estimate`
- `compare_delta_extension`

The model keywords are `wlc`, `extensible_wlc`, `fjc`, and `extensible_fjc`. A length can be supplied either as a physical length in nm or as a base count / nt count when you pass `rise_per_unit_nm` (for example `0.34` for dsDNA bases or `0.63` for ssDNA nucleotides).

## Quick usage

```python
import numpy as np
from optical_trap_dna import (
    single_polymer_extension,
    full_length_curve,
    target_curve,
    distance_estimate,
)

force = np.linspace(0.1, 20.0, 200)

# dsDNA example: 0.34 nm/bp, Lp = 50 nm, S = 1200 pN, additive eWLC

ds_params = {
    "model": "extensible_wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}

x_ds = single_polymer_extension(force, 13250.0, 50.0, 1200.0,
                               model="extensible_wlc", rise_per_unit_nm=0.34)
print(x_ds[:3])

# ssDNA example: 0.63 nm/nt, Lp = 1.2 nm, S = 800 pN, WLC
ss_params = {
    "model": "wlc",
    "persistence_length_nm": 1.2,
    "stretch_modulus_pn": 800.0,
    "rise_per_unit_nm": 0.63,
}

x_ss = single_polymer_extension(force, 30.0, 1.2, 800.0,
                               model="wlc", rise_per_unit_nm=0.63)

# Looped construct: dsDNA region + loop arm
full = full_length_curve(force, 13250.0, 300.0, ds_params, ds_params)
result = target_curve(force, 13250.0, 300.0, 150.0, ds_params, ds_params, ds_params)

d = distance_estimate(force, 13250.0, 300.0, 150.0, ds_params, ds_params, ds_params)
print(result["extension_nm"][-1], d[-1])
```

## Worked examples

### 1. dsDNA single-polymer curve

The usual dsDNA defaults are a contour rise of `0.34 nm/bp`, persistence length `50 nm`, stretch modulus `1200 pN`, and model `wlc` or `extensible_wlc` for backbone stretch. A typical call is:

```python
force = np.linspace(0.1, 50.0, 200)
params = {
    "model": "extensible_wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}

x = single_polymer_extension(force, 13250.0, 50.0, 1200.0,
                            model="extensible_wlc", rise_per_unit_nm=0.34)
```

The additive eWLC form used here is `x(F) = x_wlc(F) + L * F / S`, with the usual dsDNA defaults above.

### 2. ssDNA single-polymer curve

For ssDNA, set the rise to `0.63 nm/nt` and choose a short persistence length, e.g. `1.0`–`1.5 nm`, with `S = 800 pN` and WLC as a reasonable starting model:

```python
force = np.linspace(0.1, 20.0, 200)
ss_params = {
    "model": "wlc",
    "persistence_length_nm": 1.2,
    "stretch_modulus_pn": 800.0,
    "rise_per_unit_nm": 0.63,
}

x_ss = single_polymer_extension(force, 30.0, 1.2, 800.0,
                               model="wlc", rise_per_unit_nm=0.63)
```

This is a common short-ssDNA parameter range; the default `_DEFAULT_SSDNA` values are a convenience only for `dna_type="ssdna"` and are not automatically applied without an explicit call.

### 3. Looped construct (full-length and 0T curves)

The construct helpers build a dsDNA serial arm plus a looped region:

```python
force = np.linspace(0.1, 20.0, 200)

ds_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}
loop_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}

x_full = full_length_curve(force, 13250.0, 300.0, ds_params, loop_params)
x_zero = zero_target_curve(force, 13250.0, ds_params)
```

The `0T` curve is the dsDNA region alone. The full-length curve is dsDNA plus loop in series.

### 4. Target curve with force split

The target curve adds a target segment in parallel with the loop. The helper returns the target and loop force split, the shared parallel extension, and the full construct extension:

```python
force = np.linspace(0.1, 20.0, 200)

ds_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}
target_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}
loop_params = {
    "model": "wlc",
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "rise_per_unit_nm": 0.34,
}

result = target_curve(force, 13250.0, 300.0, 150.0, ds_params, target_params, loop_params)
print(result["force_target_pn"])
print(result["force_loop_pn"])
print(result["parallel_extension_nm"])
```

The target and loop forces satisfy `F_target + F_loop = F_total`.

## Parameter dictionary keys and defaults

The construct helpers accept parameter dictionaries with keys that are interpreted by `optical_trap_dna.py`.

| Key | Default (if omitted) | Unit | Notes |
|---|---:|---|---|
| `rise_per_unit_nm` | `1.0` | nm per bp or nt or `1.0` for already-nm lengths | For dsDNA use `0.34`; for ssDNA use `0.63`. |
| `persistence_length_nm` | `50.0` | nm | The dsDNA value; ssDNA must set this explicitly unless using `dna_type="ssdna"`. |
| `stretch_modulus_pn` | `1200.0` | pN | A positive stretch modulus is required even for inextensible models. |
| `model` | `"wlc"` | n/a | One of `wlc`, `extensible_wlc`, `fjc`, `extensible_fjc`. |
| `temperature_k` | `298.15` | K | Optional thermal override. |
| `kbt_pn_nm` | `4.114` | pN·nm | Optional direct override of thermal energy. |

For the single-polymer convenience functions, missing values fall back silently to dsDNA defaults unless `dna_type="ssdna"` is passed. The `_DEFAULT_SSDNA` values are only used by those single-polymer calls; they are not read by the construct helpers.

## Pitfalls and cautions

- The model keywords are `wlc`, `extensible_wlc`, `fjc`, and `extensible_fjc`.
- A length can be provided in bp or nt by passing `rise_per_unit_nm`; by default, a raw number is treated as nanometers.
- Missing parameter-dictionary keys fall back to dsDNA-like defaults (`persistence_length_nm = 50.0`, `stretch_modulus_pn = 1200.0`, `model = "wlc"`, `rise_per_unit_nm = 1.0`). For ssDNA, every key must be supplied explicitly unless you are calling the single-polymer helper with `dna_type="ssdna"`.
- The FJC Kuhn length is fixed at `2 * Lp`; the model does not treat this as free parameter.
- A stretch modulus is required even for inextensible models because the APIs validate `stretch_modulus_pn > 0`.
- The parallel force split solves for equal target and loop extensions by default. If the root-finding search fails, the code falls back to an equal split without warning.
- The model-data residual is not itself an estimate of the distance `d`; a measured delta extension must be compared with a model prediction, and the fitted `d` arises from the difference between two model-based delta-extension curves.

To turn a measured delta extension into a distance estimate, compare the measured `Δ_0T` or `Δ_target` curve with the model and then compute:

```python
# measured_delta_nm: observed delta extension in nm
# model_delta_nm: predicted delta extension for the same force points
comparison = compare_delta_extension(measured_delta_nm, force, model_delta_nm)

# distance estimate from the model difference between two curves
# d = x_target - x_0T = Δ_0T - Δ_target
estimated_distance = distance_estimate(force, ds_length, target_length, loop_length,
                                      ds_params, target_params, loop_params)
```

The model-data residual is a goodness-of-fit diagnostic; it is not the same as a distance estimate.

## Model schematic and citations

Figure 1 shows the three curves the toolkit computes, drawn as springs. The 0T curve is the dsDNA region alone. The full-length curve is the dsDNA region in series with the loop region. The target curve is the dsDNA region in series with a block in which the loop and the target sit in parallel, so they share one extension and the force splits between them. Segments in series carry the same force and their extensions add. The delta extensions are Δ_0T = x_full − x_0T and Δ_target = x_full − x_target, and the distance estimate is Δ_0T − Δ_target = x_target − x_0T.

![DNA FE toolkit model schematic](DNA_FE_toolkit_model_schematic.png)

**Figure 1.** Spring schematic of the DNA force-extension toolkit models: the 0T curve (dsDNA region alone), the full-length curve (dsDNA region in series with the loop), and the target curve (dsDNA region in series with the loop and target in parallel), with the delta-extension and distance equations below. A vector version is in `DNA_FE_toolkit_model_schematic.svg`.

## How to cite

When you reference the DNA force-extension toolkit, please cite Figure 1 in `docs/dna_fe_toolkit/` (`DNA_FE_toolkit_model_schematic.png` or `.svg`) and the repository issue/PR for this toolkit.
