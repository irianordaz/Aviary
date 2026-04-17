# multi_fuel

Extensions to [Aviary](https://github.com/OpenMDAO/Aviary) that allow a different CSV engine deck to be used for each mission phase, and that report per-engine-table total fuel burn as an output variable.

## Contents

| File | Purpose |
|------|---------|
| `table_builder.py` | `EngineTableBuilder`, `MultiEngineTableBuilder`, `MultiEngineFuelBurnComp` |
| `run_single_aisle.py` | Example: LSA-2 aircraft with two engine tables across three phases |

---

## Classes

### `EngineTableBuilder`

A thin wrapper around Aviary's `EngineDeck` that accepts a CSV path at construction time rather than requiring it to be set on an `AviaryValues` object manually.

```python
from multi_fuel.table_builder import EngineTableBuilder

engine = EngineTableBuilder(
    name='my_engine',
    csv_path='models/engines/turbofan_22k.csv',
)
```

Accepts both Aviary-relative paths (resolved via `get_path`) and absolute paths.

---

### `MultiEngineTableBuilder`

Selects a different `EngineTableBuilder` CSV deck for each named mission phase and tracks a fuel density per phase. All CSV files are pre-loaded at construction time. During the mission, the correct deck is dispatched based on the phase name injected into `subsystem_options` by `configure_phase_info`.

#### Constructor

```python
MultiEngineTableBuilder(
    name: str = 'multi_engine_table',
    phase_engine_map: dict = None,
)
```

- **`name`** — subsystem label; must be unique among all external subsystems passed to `load_external_subsystems`.
- **`phase_engine_map`** — mapping of phase name → `(csv_path, density_lbm_per_galUS)` tuple, or a plain CSV path string (falls back to Aviary's default density, 6.7 lbm/galUS). Using tuples allows the same engine CSV to carry a different fuel density in different phases. Uniqueness in the output array is determined by `(csv, density)` pair, so the same CSV with two different densities produces two separate output entries.

#### Usage

```python
from copy import deepcopy
from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info
from multi_fuel.table_builder import MultiEngineTableBuilder

engine = MultiEngineTableBuilder(
    name='multi_engine_table',
    phase_engine_map={
        'climb':   ('models/engines/turbofan_28k.csv', 6.7),  # Jet-A
        'cruise':  ('models/engines/turbofan_22k.csv', 6.4),  # SAF blend
        'descent': ('models/engines/turbofan_22k.csv', 3.5),  # LNG
    },
)

# Inject phase names into phase_info so the builder knows which deck to use.
# Call this before passing phase_info to AviaryProblem.
phase_info = engine.configure_phase_info(deepcopy(phase_info))

prob = AviaryProblem()
prob.load_inputs(aviary_inputs, phase_info)
prob.load_external_subsystems([engine])   # register as an engine model
prob.check_and_preprocess_inputs()
prob.build_model()
# ... add driver, design variables, objective, setup, run ...
```

#### Output variables

After the mission runs, per-engine fuel burn is available via:

```python
from aviary.variable_info.variables import Mission

fuel_mass   = prob.get_val(Mission.TOTAL_FUEL_MULTI,        units='lbm')
fuel_volume = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS')
```

Both arrays are indexed by unique `(csv, density)` pair, in order of first appearance. In the example above:

| Index | CSV | Density (lbm/galUS) | Phases contributing |
|-------|-----|---------------------|---------------------|
| 0 | `turbofan_28k.csv` | 6.7 | climb |
| 1 | `turbofan_22k.csv` | 6.4 | cruise |
| 2 | `turbofan_22k.csv` | 3.5 | descent |

Because cruise and descent use the same CSV but different densities, they appear as separate entries. Fuel burn for each phase is the change in gross mass from phase start to phase end (`mass[t=0] − mass[t=final]`). Volume is derived by dividing each mass entry by its corresponding density.

#### `configure_phase_info(phase_info, propulsion_name='propulsion')`

Injects the phase name into each flight phase's `subsystem_options` so `build_mission` can dispatch to the correct engine deck. Modifies `phase_info` in-place and returns it. Skips `pre_mission` and `post_mission` entries.

---

## Changes made to Aviary

Three Aviary source files were modified to support this extension. All changes are additive and do not affect existing behaviour.

### `aviary/variable_info/variables.py`

Added `Mission.TOTAL_FUEL_MULTI = 'mission:total_fuel_multi'` to the `Mission` class.

### `aviary/variable_info/variable_meta_data.py`

Added `add_meta_data` entries for both output variables:

| Variable | Units | Description |
|----------|-------|-------------|
| `Mission.TOTAL_FUEL_MULTI` | `lbm` | Fuel mass burned per unique `(csv, density)` pair, ordered by first appearance in `phase_engine_map`. |
| `Mission.TOTAL_FUEL_VOLUME_MULTI` | `galUS` | Fuel volume burned per unique `(csv, density)` pair, derived by dividing each mass entry by its phase density. |

### `aviary/subsystems/propulsion/engine_model.py`

Two new methods added to the `EngineModel` base class, both returning empty defaults so existing engine models are unaffected:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_post_mission_promotes_outputs` | `() → list` | Returns the `promotes_outputs` list used when adding the post-mission component to the `PostMissionGroup`. |
| `get_traj_connections` | `(regular_phases) → list[tuple]` | Returns `(src, tgt, src_indices)` tuples that `AviaryGroup.link_phases` uses to wire trajectory state timeseries into the post-mission component. |

### `aviary/core/aviary_group.py`

Two additions inside `AviaryGroup`:

1. **`add_post_mission_systems`** — after the existing loop over `self.subsystems`, a new loop calls `build_post_mission` on every entry in `self.engine_models`. If it returns a component, that component is added to `PostMissionGroup` (without explicit promotes; `PostMissionGroup.configure` handles promotion of `mission:*` and `aircraft:*` variables automatically).

2. **`link_phases`** — after the standard trajectory-to-`state_output` connections, a new loop calls `get_traj_connections` on each engine model and registers the returned connections. This is how the start and end mass of each trajectory phase reach the `MultiEngineFuelBurnComp` inputs.
