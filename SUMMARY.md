# Aviary + HyTank Integration — Work Summary

## 1. Auto-reorder on `LH2TankWeightComp` (cryo_example.py)

Added `self.options['auto_order'] = True` at the top of
`LH2TankWeightComp.setup()`. This lets OpenMDAO determine the
execution order of the `tank_weight` and `unit_conversion`
subsystems from their data connections rather than the order
they were added.

Pattern matches existing Aviary usage in
`aviary/subsystems/premission.py`,
`aviary/core/post_mission_group.py`, and the mission ODE groups
under `aviary/mission/`.

## 2. Auto-reorder on `PreMissionGroup` (aviary/core/pre_mission_group.py)

`PreMissionGroup` previously only defined `configure()`. Added a
`setup()` method that sets `self.options['auto_order'] = True`
so the core pre-mission group inherits the same automatic
ordering behavior.

## 3. Mission-level integration of HyTank `LH2Tank` (cryo_example.py)

### New imports
- `numpy as np`
- `aviary.variable_info.variables.Dynamic`
- `hytank.LH2_tank.LH2TankThermals`

### New `LH2TankMissionComp` group
A wrapper around `LH2TankThermals` (heat-leak + boil-off; the
structural weight is already produced in pre-mission, so it is
intentionally excluded here to avoid duplicate computation):

- `initialize()` declares an `int` option `num_nodes`.
- `setup()`:
  - Sets `auto_order = True`.
  - Adds `LH2TankThermals(num_nodes=nn)`, promoting
    `radius`, `length`, `N_layers`, `P_heater`,
    `m_dot_gas_out`, `T_env`, and renaming
    `m_dot_liq_out` → `Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL`.
  - Promotes outputs `m_gas`, `m_liq`, `T_gas`, `T_liq`,
    `P`, `fill_level`.
  - Sets defaults for inputs Aviary does not drive:
    `P_heater = 0 W`, `m_dot_gas_out = 0 kg/s`,
    `T_env = 273 K`.

### Sign and units
- Aviary's `FUEL_FLOW_RATE_TOTAL` is declared in `lbm/h`
  with consumption positive.
- HyTank's `m_dot_liq_out` is `kg/s`, also positive on
  extraction.
- OpenMDAO performs the `lbm/h` ↔ `kg/s` conversion
  automatically through the declared input units.

### `LH2TankBuilder` additions
- `build_mission(num_nodes, aviary_inputs, user_options=None, subsystem_options=None)`
  returns `LH2TankMissionComp(num_nodes=num_nodes)`.
- `mission_inputs()` restricts mission promotions to the
  explicit list above (instead of the default `['*']`).
- `mission_outputs()` exposes the thermal/boil-off state
  variables.

## 4. Changes inside the `aviary/` folder

### `aviary/core/pre_mission_group.py`
- Added a `setup()` method that sets
  `self.options['auto_order'] = True`. `PreMissionGroup`
  previously only defined `configure()`, so there was nowhere
  to enable auto-ordering. The new `setup()` brings it in line
  with `PostMissionGroup` and the mission ODE groups.

No other files inside `aviary/` were modified.

## 5. Changes inside the `HyTank/` folder

All HyTank edits unify the units declaration of the `N_layers`
input to `"unitless"` so it lines up with the Aviary
`Aircraft.Fuel.LH2Tank.N_LAYERS` metadata and silences an
OpenMDAO `UnitsWarning` raised during connection.

### `HyTank/hytank/weight.py`
- Line 455: `self.add_input("N_layers")` →
  `self.add_input("N_layers", units="unitless")`. Previously the
  input had no units (defaulting to `None`), which mismatched
  the auto-IVC's `'unitless'` units when promoted through the
  Aviary data hierarchy.

### `HyTank/hytank/heat_leak.py`
- Line 171: `self.add_input("N_layers", val=20, units=None)` →
  `self.add_input("N_layers", val=20, units="unitless")`. Same
  rationale as `weight.py`.

### `HyTank/hytank/LH2_tank.py`
- Line 215 (`LH2Tank.setup`) and line 342
  (`LH2TankThermals.setup`):
  `self.set_input_defaults("N_layers", 20)` →
  `self.set_input_defaults("N_layers", 20, units="unitless")`.
  Without the explicit units, the group-level defaults still
  registered as no-unit, so the auto-IVC could end up with
  `'unitless'` from one promoted input and no-units from
  another.

No other files inside `HyTank/` were modified.

## 6. LNG (methane) support in HyTank, parallel to LH2

Added liquefied methane as a second propellant option without
disturbing the existing LH2 code path. Approach: parallel
implementation (new data, new property class, new wrapper) with
an OpenMDAO `propellant` option (`'LH2'` default, `'LNG'` opt-in)
threaded through the physics components.

### New files
- `HyTank/hytank/CH4_property_data/__init__.py`
- `HyTank/hytank/CH4_property_data/data_parser.py` — copy of the H2
  parser; reads saturated and per-bar tab-separated property files
  in this directory.
- `HyTank/hytank/CH4_property_data/generate_from_coolprop.py` —
  one-shot script that pulls methane property tables from
  CoolProp and writes them in the NIST schema. Run via
  `pixi run python HyTank/hytank/CH4_property_data/generate_from_coolprop.py`.
- `HyTank/hytank/CH4_property_data/saturated_properties.txt` +
  seven `<P>_bar_properties.txt` files (1, 2, 5, 10, 20, 30, 40 bar).
- `HyTank/hytank/CH4_properties.py` — `MethaneProperties(HydrogenProperties)`.
  Reuses the H2 surrogate code via subclassing; overrides
  `_data_subdir`, `_get_sat_property`, `_get_property`, and
  `MOLEC_WEIGHT`.
- `HyTank/hytank/LNG_tank.py` — `LNGTank(LH2Tank)` and
  `LNGTankThermals(LH2TankThermals)` convenience wrappers that
  shift initial-condition defaults (ullage 118 K @ 1.5 bar, liquid
  113 K) and set `propellant='LNG'`.
- `HyTank/hytank/tests/test_CH4_properties.py`,
  `HyTank/hytank/tests/test_LNG_tank.py` — sanity + build-and-run
  tests (5 + 1 new tests, all passing).

### Edited files
- `pixi.toml` — added `coolprop>=7.2.0,<8` to `[pypi-dependencies]`.
- `HyTank/hytank/utilities/constants.py` — added
  `MOLEC_WEIGHT_CH4 = 16.043e-3`.
- `HyTank/hytank/H2_properties.py`:
  - Added `MOLEC_WEIGHT = MOLEC_WEIGHT_H2` class attribute.
  - Refactored `__init__` to read data through the three new class
    hooks `_data_subdir`, `_get_sat_property`, `_get_property` so
    `MethaneProperties` can subclass cleanly without duplicating
    the surrogate-training body.
- `HyTank/hytank/H2_properties_MendezRamos.py` — exposed
  `MOLEC_WEIGHT = MOLEC_WEIGHT_H2` at module scope so the
  complex-step derivative tests (which swap `self.H2` for this
  module) keep working after the constant migrated onto the
  property class.
- `HyTank/hytank/boil_off.py` (the biggest edit):
  - Replaced the eager module-level `H2_prop = HydrogenProperties()`
    with a lazy `get_propellant(name)` cache that returns either a
    `HydrogenProperties` or `MethaneProperties` instance.
  - Added a `propellant` OpenMDAO option (default `'LH2'`,
    values `('LH2', 'LNG')`) on `BoilOff`, `FullODE`,
    `LH2BoilOffODE`, and `InitialTankStateModification`.
  - Replaced direct `H2_prop.*` calls with `self.H2.*` (lines
    283, 298, 299, 460 in the original file).
  - Replaced 19 occurrences of `MOLEC_WEIGHT_H2` with
    `self.H2.MOLEC_WEIGHT` inside `LH2BoilOffODE.compute` /
    `compute_partials`.
  - Moved `self.H2 = ...` from `InitialTankStateModification.__init__`
    into `setup` so it can read the (post-init) option.
  - Made `T_gas` / `T_liq` output bounds in
    `InitialTankStateModification.setup` propellant-aware
    (LH2: [14, 33] / [18, 150] K; LNG: [88, 120] / [88, 200] K).
    Without this, methane initial states (~110-118 K) get
    immediately clamped to LH2 ranges and Newton diverges.
- `HyTank/hytank/LH2_tank.py` — added `propellant` option to
  `LH2Tank.initialize` and `LH2TankThermals.initialize` and
  forwarded it to the child subsystems they construct.
- `HyTank/hytank/__init__.py` — re-exported `LNGTank` and
  `LNGTankThermals` alongside the existing LH2 classes.

### Why the parallel approach works
- `HydrogenProperties`'s public method names (`lh2_P`, `gh2_rho`,
  `sat_gh2_T`, ...) live in the surrogate dictionaries as
  name-bound keys, not as physics names. `MethaneProperties`
  inherits those same methods but populates the dictionaries from
  the methane data files. Every existing call site in `boil_off.py`
  stays valid — only the data behind it changes.
- Surrogate construction is expensive; the lazy `get_propellant`
  cache shares a single `HydrogenProperties` (and a single
  `MethaneProperties`) instance across every component that picks
  the same propellant.
- LH2 path is untouched: `propellant='LH2'` is the default, the
  same `.pkl` caches are used, and the full LH2 test suite (43
  tests across `test_boil_off`, `test_LH2_tank`, `test_heat_leak`,
  `test_weight`) still passes.

### Verification
- LH2 tests (backward compat): 43/43 pass.
- LNG tests (new): 6/6 pass.
- End-to-end LNG smoke run (1 m × 4 m tank, 1 hr cruise, 0.05 kg/s
  draw) burns exactly 180 kg of LNG (conservation check ✓),
  pressure drops from 1.50 → 1.12 bar, fill drops 95% → 92.4%,
  liquid T stable at 113 K.

## 7. Fixed NaN/inf error at `num_nodes=37` (BDF3 stability)

HyTank's BDF3 integrator has a dt-dependent stability region. When
the trajectory produced 36 total nodes (rounded up to odd → 37),
the resulting `dt ≈ 750 s` for a ~27 000 s mission pushed `h×λ`
outside the stable region and Newton diverged after 4 iterations
with `inf`/`NaN` residuals.

### Root cause
- BDF3 stability depends on the product of the step size and the
  ODE Jacobian eigenvalue. For LNG boil-off at typical flow rates,
  `dt ≈ 750 s` is unstable; `dt ≈ 2 700 s` (11 nodes) is robustly
  stable.
- Systematic tests over `nn=11–73` and flow rates `0–1.0 kg/s`
  confirmed: 11–35 nodes mostly stable, 37–61 mostly unstable,
  63+ mostly stable again — irregular, flow-rate-dependent pattern.

### Fix
Decoupled HyTank's analysis grid from the Dymos trajectory
resolution. Added a `hytank_num_nodes` parameter to
`CryoTankBuilder` (default `11`). The `_MissionFuelFlowAssembler`
still reads the full trajectory at whatever resolution Dymos
produces, but resamples onto the fixed 11-node grid before feeding
HyTank. This keeps `dt ≈ 2 700 s` regardless of trajectory node
count, while `phase_mission_bus_lengths` continues to supply the
true Dymos node counts for the bus connections.

## 8. Arbitrary phase names and counts

Removed all hardcoded phase lists (`_PHASES`, `_PHASE_TIME_DEFAULTS`)
from `cryo_example_lng.py` (and subsequently `cryo_example.py`).

### Changes
- `_MissionFuelFlowAssembler` — phases and bus lengths supplied
  as component options (`phase_names`, `bus_lengths`) set at
  construction time by the builder; adds per-phase inputs
  dynamically in `setup()`.
- `CryoTankPostMissionComp` — same `phase_names`/`bus_lengths`
  options; sets per-phase `set_input_defaults` from a helper.
- `_phase_time_defaults(phase_names, bus_lengths)` — generates
  non-degenerate sequential time arrays (100 s/node, 100 s gap
  between phases) for each phase before bus connections are live.
- `CryoTankBuilder.build_post_mission` — reads
  `phase_mission_bus_lengths` (actual Dymos node counts) when
  provided by Aviary; falls back to `mission_info.keys()` or the
  three-phase default if called outside the full Aviary flow.
- `CryoTankBuilder.get_post_mission_bus_variables` — iterates
  `mission_info.keys()` so any rename or addition in `phase_info`
  is reflected automatically.

Adding a fourth phase (e.g. `reserve_cruise`) to `phase_info`
requires zero changes to the builder or component code.

## 9. Shared infrastructure extracted to `cryo_builder.py`

`cryo_example_lng.py` and `cryo_example.py` (LH2) had identical
builder/component structure differing only in propellant-specific
details. Consolidated into a single generic module.

### New file: `cryo_builder.py`
Public exports:

| Name | Type | Purpose |
|---|---|---|
| `KG_TO_LBM` | constant | unit conversion |
| `_phase_time_defaults` | function | pre-connection time defaults |
| `_TankWeightToFuelSystemMass` | `ExplicitComponent` | kg → lbm, maps to `FUEL_SYSTEM_MASS` |
| `_MissionFuelFlowAssembler` | `ExplicitComponent` | concatenate + resample per-phase fuel-flow |
| `CryoTankWeightComp` | `Group` | pre-mission tank weight wrapper |
| `CryoTankPostMissionComp` | `Group` | post-mission HyTank thermal analysis |
| `CryoTankBuilder` | `SubsystemBuilder` | Aviary external subsystem builder |

`CryoTankBuilder.__init__` signature:
```python
CryoTankBuilder(
    name,               # Aviary subsystem name
    meta_data,          # extended metadata dict
    tank_vars,          # dict: RADIUS/LENGTH/… → Aviary var strings
    thermals_class,     # e.g. LH2TankThermals or LNGTankThermals
    thermals_kwargs=None,   # extra kwargs forwarded to thermals_class
    t_env_default=300.0,    # default env temperature (K)
    hytank_num_nodes=11,    # odd HyTank BDF3 node count
)
```

### `cryo_example_lng.py` — rewritten as thin wrapper
Only LNG-specific content remains:
- `Aircraft.Fuel.LNGTank` variable extension + 6 `av.add_meta_data`
  calls.
- `_LNG_TANK_VARS` mapping dict.
- Main script: `CryoTankBuilder(name='lng_tank', ...,
  thermals_class=LNGTankThermals, t_env_default=300.0)`.
- LNG-specific mass initial guesses for the trajectory states.

### `cryo_example.py` → `cryo_example_lh2.py` (renamed + rewritten)
Parallel thin wrapper for LH2:
- `Aircraft.Fuel.LH2Tank` variable extension + metadata.
- `_LH2_TANK_VARS` mapping dict.
- Main script: `CryoTankBuilder(name='lh2_tank', ...,
  thermals_class=LH2TankThermals,
  thermals_kwargs={'ullage_T_init': 22.0}, t_env_default=273.0)`.

`cryo_example.py` deleted.

## Files touched
- `cryo_example.py` → deleted
- `cryo_example_lh2.py` (renamed + rewritten from `cryo_example.py`)
- `cryo_example_lng.py` (rewritten)
- `cryo_builder.py` (new)
- `aviary/core/pre_mission_group.py`
- `pixi.toml`
- `HyTank/hytank/weight.py`
- `HyTank/hytank/heat_leak.py`
- `HyTank/hytank/LH2_tank.py`
- `HyTank/hytank/__init__.py`
- `HyTank/hytank/utilities/constants.py`
- `HyTank/hytank/H2_properties.py`
- `HyTank/hytank/H2_properties_MendezRamos.py`
- `HyTank/hytank/boil_off.py`
- `HyTank/hytank/CH4_properties.py` (new)
- `HyTank/hytank/LNG_tank.py` (new)
- `HyTank/hytank/CH4_property_data/` (new directory)
- `HyTank/hytank/tests/test_CH4_properties.py` (new)
- `HyTank/hytank/tests/test_LNG_tank.py` (new)
- `SUMMARY.md` (this file)
