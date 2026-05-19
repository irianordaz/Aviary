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

## Files touched
- `cryo_example.py`
- `aviary/core/pre_mission_group.py`
- `HyTank/hytank/weight.py`
- `HyTank/hytank/heat_leak.py`
- `HyTank/hytank/LH2_tank.py`
- `SUMMARY.md` (this file)
