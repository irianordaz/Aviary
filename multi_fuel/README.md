# Phase-Based Multi-Fuel Engine Capability

An [Aviary](https://github.com/OpenMDAO/Aviary) extension that enables using different CSV engine decks with distinct fuel densities for each mission phase, and reporting per-engine total fuel burn as OpenMDAO output variables.

## Overview

Aviary's standard engine model uses a single CSV performance deck across all flight phases. The `multi_fuel` extension introduces `MultiEngineTableBuilder`, which dispatches a different `EngineDeck` for each named phase (climb, cruise, descent, etc.) and tracks a fuel density per phase.

After the mission runs, per-engine fuel burn is summed and exposed as:
- `Mission.TOTAL_FUEL_MULTI` (mass, lbm)
- `Mission.TOTAL_FUEL_VOLUME_MULTI` (volume, galUS)

Both arrays are indexed by unique `(csv, density)` pair in order of first appearance. The same CSV file used with two different fuel densities produces two separate output entries.

## Quick Start

```python
from copy import deepcopy
from multi_fuel.table_builder import MultiEngineTableBuilder

# Define phase-to-engine mappings with fuel densities
engine = MultiEngineTableBuilder(
    phase_engine_map={
        'climb':   ('models/engines/turbofan_28k.csv', 6.7),  # Jet-A
        'cruise':  ('models/engines/turbofan_22k.csv', 6.4),  # SAF blend
        'descent': ('models/engines/turbofan_22k.csv', 3.5),  # LNG
    },
)

# Configure phase info before passing to AviaryProblem
phase_info = engine.configure_phase_info(deepcopy(phase_info))

# Use with AviaryProblem
prob = AviaryProblem()
prob.load_inputs(aviary_inputs, phase_info)
prob.load_external_subsystems([engine])
prob.check_and_preprocess_inputs()
prob.build_model()
prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
prob.add_design_variables()
prob.add_objective()
prob.setup()
prob.run_aviary_problem()

# Access results
from aviary.variable_info.variables import Mission
fuel_mass = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')
fuel_volume = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS')
```

## File Structure

| File | Description |
|------|-------------|
| `table_builder.py` | Core implementation: `MultiEngineFuelBurnComp`, `EngineTableBuilder`, `MultiEngineTableBuilder` |
| `run_single_aisle.py` | Complete example: LSA-2 aircraft with multi-fuel engine decks across three phases |

## Classes

### `EngineTableBuilder`

A thin wrapper around Aviary's `EngineDeck` that accepts a CSV path directly at construction time.

```python
from multi_fuel.table_builder import EngineTableBuilder

engine = EngineTableBuilder(
    name='my_engine',
    csv_path='models/engines/turbofan_22k.csv',
)
```

Accepts both Aviary-relative paths (resolved via `get_path`) and absolute paths.

### `MultiEngineTableBuilder`

The primary entry point for multi-fuel simulation. Manages a collection of engine models, each associated with a specific mission phase.

#### Constructor

```python
MultiEngineTableBuilder(
    name: str = 'multi_engine_table',
    phase_engine_map: dict = None,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Subsystem label. Must be unique among all external subsystems passed to `load_external_subsystems`. |
| `phase_engine_map` | `dict` | Mapping of phase name → `(csv_path, density_lbm_per_galUS)` tuple, or a plain CSV path string (falls back to Aviary's default density, 6.7 lbm/galUS). |

**Key Behaviors:**

- String values in `phase_engine_map` are automatically converted to `(csv, density)` tuples using the default fuel density
- Uniqueness in output arrays is determined by `(csv, density)` pair — the same CSV with two different densities produces two separate output entries
- All CSV files are pre-loaded at construction time
- Phase name injection into `subsystem_options` is handled by `configure_phase_info`

#### Methods

**`configure_phase_info(phase_info, propulsion_name='propulsion')`**

Injects the phase name into each flight phase's `subsystem_options` so `build_mission` can dispatch to the correct engine deck.

- Modifies `phase_info` in-place and returns it
- Skips `pre_mission` and `post_mission` entries
- **Must be called before `AviaryProblem.load_inputs`**

**`get_post_mission_promotes_outputs()`**

Returns the list of output variables to promote from the post-mission subsystem. Returns `[Mission.TOTAL_FUEL_MULTI]`.

**`build_post_mission(...)`**

Creates and returns a `MultiEngineFuelBurnComp` instance that computes total per-engine fuel burn from phase start/end mass inputs.

**`get_traj_connections(regular_phases)`**

Returns trajectory variable connections for wiring mass timeseries into the fuel burn component. Returns a list of `(src, tgt, src_indices)` tuples.

**`build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)`**

Selects and builds the engine deck system for the current mission phase based on `subsystem_options['phase_name']`.

**`build_pre_mission(aviary_inputs, subsystem_options=None)`**

Delegates pre-mission sizing to the first configured engine deck.

## Output Variables

After the mission runs, per-engine fuel burn is available as mass and volume arrays:

```python
from aviary.variable_info.variables import Mission

# Mass array (lbm)
fuel_mass = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')

# Volume array (galUS)
fuel_volume = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS')
```

Both arrays are indexed by unique `(csv, density)` pair in order of first appearance.

**Example Index Mapping:**

| Index | CSV | Density (lbm/galUS) | Phases Contributing |
|-------|-----|---------------------|---------------------|
| 0 | `turbofan_28k.csv` | 6.7 | climb |
| 1 | `turbofan_22k.csv` | 6.4 | cruise |
| 2 | `turbofan_22k.csv` | 3.5 | descent |

**Important Notes:**

- Array length equals the number of unique `(csv, density)` pairs in `phase_engine_map`
- Fuel burn for each phase is computed as `mass[t=0] - mass[t=final]` (change in gross mass)
- Volume is derived by dividing mass by the phase's fuel density
- Only trajectory phases are included (taxi and takeoff fuel is modeled separately)

## Aviary Integration

All changes to Aviary are additive and do not affect existing engine models or mission setups.

### Modified Files

**`aviary/variable_info/variables.py`**

Added `TOTAL_FUEL_MULTI = 'mission:total_fuel_multi'` to the `Mission` class.

**`aviary/variable_info/variable_meta_data.py`**

Added metadata entries for both output variables with units (`lbm` and `galUS`) and descriptions.

**`aviary/subsystems/propulsion/engine_model.py`**

Extended `EngineModel` base class with two new methods returning empty defaults:

| Method | Default Return | Purpose |
|--------|----------------|---------|
| `get_post_mission_promotes_outputs()` | `[]` | Returns output variable list for post-mission component |
| `get_traj_connections(regular_phases)` | `[]` | Returns trajectory connection specs |

**`aviary/core/aviary_group.py`**

Added two internal methods:

1. **`add_post_mission_systems`** — Iterates `self.engine_models` and adds post-mission components to `PostMissionGroup`
2. **`link_phases`** — Wires trajectory mass timeseries into post-mission component inputs

## Running the Example

Execute the complete LSA-2 example with multi-fuel engine decks:

```bash
pixi run python multi_fuel/run_single_aisle.py
```

Expected output:

```
Success: True
Total fuel (lbm): [19536.72787796]
Per-engine fuel (lbm): [ 6149.880128   12727.84774995]
Per-engine fuel (galUS): [ 917.89255642 1988.72621093]
```

## When to Use This Extension

Use `multi_fuel` when you need to:

- Simulate aircraft operations with different fuel types across mission phases (Jet-A, SAF blend, LNG, etc.)
- Model engine performance variations between different engine decks for different phases
- Track fuel consumption separately by fuel type for reporting and analysis
- Optimize mission profiles with multi-fuel constraints

## Limitations

- Only supports CSV-based engine decks (not parameterized engine models)
- Fuel density is assumed constant within each phase
- Pre-mission and post-mission phases use the first configured engine deck
- Taxi and takeoff fuel burn is not included in the multi-fuel output arrays

## See Also

- [Aviary Documentation](https://openmdao.org/aviary)
- `multi_fuel/table_builder.py` — Source code with detailed docstrings
- `multi_fuel/docs/index.html` — Interactive documentation with examples
- `aviary/subsystems/propulsion/engine_deck.py` — Base EngineDeck implementation
