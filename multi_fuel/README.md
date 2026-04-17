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

Selects a different `EngineTableBuilder` CSV deck for each named mission phase. All CSV files are pre-loaded at construction time. During the mission, the correct deck is dispatched based on the phase name injected into `subsystem_options` by `configure_phase_info`.

#### Constructor

```python
MultiEngineTableBuilder(
    name: str = 'multi_engine_table',
    phase_engine_map: dict = None,
)
```

- **`name`** — subsystem label; must be unique among all external subsystems passed to `load_external_subsystems`.
- **`phase_engine_map`** — mapping of phase name → CSV path. Multiple phases may share the same CSV; phases that share a CSV are summed together in `Mission.TOTAL_FUEL_MULTI`. The order of first appearance determines the index order in that output array.

#### Usage

```python
from copy import deepcopy
from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info
from multi_fuel.table_builder import MultiEngineTableBuilder

engine = MultiEngineTableBuilder(
    name='multi_engine_table',
    phase_engine_map={
        'climb':   'models/engines/turbofan_28k.csv',
        'cruise':  'models/engines/turbofan_22k.csv',
        'descent': 'models/engines/turbofan_22k.csv',
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

#### Output variable: `Mission.TOTAL_FUEL_MULTI`

After the mission runs, per-engine fuel burn is available via:

```python
from aviary.variable_info.variables import Mission

fuel_per_engine = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')
# fuel_per_engine[0] → total fuel burned by the first unique CSV in phase_engine_map
# fuel_per_engine[1] → total fuel burned by the second unique CSV, etc.
```

The array length equals the number of **unique** CSV paths in `phase_engine_map`, ordered by first appearance. In the example above:

| Index | CSV | Phases contributing |
|-------|-----|---------------------|
| 0 | `turbofan_28k.csv` | climb |
| 1 | `turbofan_22k.csv` | cruise + descent |

Fuel burn for each phase is computed as the change in gross mass from the start to the end of that phase (`mass[t=0] − mass[t=final]`), then summed across phases that share the same engine CSV.

#### `configure_phase_info(phase_info, propulsion_name='propulsion')`

Injects the phase name into each flight phase's `subsystem_options` so `build_mission` can dispatch to the correct engine deck. Modifies `phase_info` in-place and returns it. Skips `pre_mission` and `post_mission` entries.

---

## Changes made to Aviary

Three Aviary source files were modified to support this extension. All changes are additive and do not affect existing behaviour.

### `aviary/variable_info/variables.py`

Added `Mission.TOTAL_FUEL_MULTI = 'mission:total_fuel_multi'` to the `Mission` class.

### `aviary/variable_info/variable_meta_data.py`

Added an `add_meta_data` entry for `Mission.TOTAL_FUEL_MULTI`:

- **units**: `lbm`
- **description**: total fuel burned per unique engine table, ordered by first appearance in `phase_engine_map`. Array length equals the number of unique engine CSVs.

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
