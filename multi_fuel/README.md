# Phase-Based Multi-Fuel Engine Capability

An [Aviary](https://github.com/OpenMDAO/Aviary) extension that lets each mission phase use its own CSV engine deck and its own fuel density, and reports per-fuel total mass and volume as OpenMDAO outputs — **without modifying any code in the `aviary/` folder**.

## Overview

Aviary's stock propulsion builder (`CorePropulsionBuilder`) always uses a single `EngineDeck` in every phase. The `multi_fuel` extension drops in a phase-aware replacement that picks the engine registered for the active phase at ODE-build time.

Two pieces make this work:

1. **`MultiPhasePropulsionBuilder`** — a subclass of `CorePropulsionBuilder` whose `build_mission` reads `subsystem_options['phase_name']` and builds a `PropulsionMission` around that phase's engine.
2. **`MultiEngineTableBuilder`** — a `SubsystemBuilder` that holds the `{phase: (csv, density)}` map, exposes helpers for installing the propulsion swap and wiring the post-mission fuel accounting component, and contributes a component that aggregates fuel burn per unique `(csv, density)` pair.

After the mission runs, per-fuel totals are exposed as module-level variables:

- `multi_fuel.table_builder.TOTAL_FUEL_MULTI` → `'mission:total_fuel_multi'` (mass, lbm)
- `multi_fuel.table_builder.TOTAL_FUEL_VOLUME_MULTI` → `'mission:total_fuel_volume_multi'` (volume, galUS)

Both arrays are indexed by unique `(csv, density)` pair in order of first appearance. The same CSV used with two different densities produces two separate output entries.

## Quick Start

```python
from copy import deepcopy

from aviary.core.aviary_problem import AviaryProblem
from multi_fuel.table_builder import (
    TOTAL_FUEL_MULTI,
    TOTAL_FUEL_VOLUME_MULTI,
    MultiEngineTableBuilder,
)

engine = MultiEngineTableBuilder(
    phase_engine_map={
        'climb':   ('multi_fuel/engines/turbofan_28k.csv', 6.7),  # Jet-A
        'cruise':  ('multi_fuel/engines/turbofan_22k.csv', 6.4),  # SAF blend
        'descent': ('multi_fuel/engines/turbofan_22k.csv', 3.5),  # LNG
    },
)

# Tag each phase so the propulsion swap can dispatch per phase.
phase_info = engine.configure_phase_info(deepcopy(phase_info))

prob = AviaryProblem()
prob.load_inputs(aviary_inputs, phase_info)

# The default engine satisfies Aviary's initial preprocessing pass; the
# builder itself contributes the post-mission fuel accounting component.
prob.load_external_subsystems([engine.get_default_engine(), engine])

prob.check_and_preprocess_inputs()

# Swap Aviary's CorePropulsionBuilder for a phase-aware one so each phase's
# ODE is built around *that* phase's engine (and its fuel-flow drives mass).
engine.install_propulsion(prob.model)

prob.build_model()

# Connect each phase's mass timeseries into the post-mission fuel-burn component.
engine.wire_trajectory(prob.model)

prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
prob.add_design_variables()
prob.add_objective()
prob.setup()
prob.run_aviary_problem()

fuel_mass = prob.get_val(TOTAL_FUEL_MULTI, units='lbm')
fuel_volume = prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS')
```

The required call order is:

```
configure_phase_info → load_inputs → load_external_subsystems
  → check_and_preprocess_inputs → install_propulsion
  → build_model → wire_trajectory → setup → run_aviary_problem
```

## File Structure

| File | Description |
|------|-------------|
| `table_builder.py` | Core implementation: `MultiEngineFuelBurnComp`, `EngineTableBuilder`, `MultiPhasePropulsionBuilder`, `MultiEngineTableBuilder` |
| `run_single_aisle.py` | Complete example: LSA-2 aircraft with multi-fuel engine decks across climb/cruise/descent |
| `test/test_table_builder.py` | Unit tests for the per-phase engine assignment, density handling, propulsion dispatch, and post-mission accounting |
| `test/test_benchmark.py` | End-to-end integration test: builds the full `AviaryProblem` and verifies each phase's ODE contains its assigned engine |
| `engines/*.csv` | Local engine decks referenced by the example and tests |

## Classes

### `EngineTableBuilder`

Thin wrapper around Aviary's `EngineDeck` that accepts a CSV path and an optional `AviaryValues` options container at construction time.

```python
from multi_fuel.table_builder import EngineTableBuilder

engine = EngineTableBuilder(
    name='my_engine',
    csv_path='multi_fuel/engines/turbofan_22k.csv',
)
```

Paths may be Aviary-relative (resolved via `get_path`) or absolute.

### `MultiPhasePropulsionBuilder`

Phase-aware subclass of `aviary.subsystems.propulsion.propulsion_builder.CorePropulsionBuilder`. Its `build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)` reads the active phase name from `subsystem_options['phase_name']`, looks up the engine model for that phase in its `phase_engines` dict, and returns a `PropulsionMission` built around that single engine.

When the phase name is missing or not in `phase_engines`, it falls back to `default_engine` — this keeps a later-added reserve phase from breaking the run.

You will not normally instantiate this class directly. `MultiEngineTableBuilder.install_propulsion` swaps it into `AviaryGroup.subsystems` in place of the default `CorePropulsionBuilder`.

### `MultiEngineTableBuilder`

Primary entry point. Holds the per-phase engine map and coordinates the three integration points with Aviary:

#### Constructor

```python
MultiEngineTableBuilder(
    name: str = 'multi_engine_table',
    phase_engine_map: dict = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Subsystem label. Must be unique among all external subsystems passed to `load_external_subsystems`. |
| `phase_engine_map` | `dict` | Mapping of phase name → `(csv_path, density_lbm_per_galUS)` tuple, or a plain CSV path string (falls back to Aviary's default fuel density). |

Key behaviors:

- String values in `phase_engine_map` are normalized to `(csv, default_density)` tuples at construction time.
- One `EngineTableBuilder` is created per phase at construction time, named `{builder.name}_{phase}`, with `Aircraft.Fuel.DENSITY` set from the per-phase density.
- Uniqueness in the output arrays is determined by `(csv, density)` pair — the same CSV with two different densities produces two separate output entries.

#### `configure_phase_info(phase_info, propulsion_name='propulsion')`

Injects the phase name into each flight phase's `subsystem_options` under the propulsion subsystem's key, so that `MultiPhasePropulsionBuilder.build_mission` can look it up. Modifies `phase_info` in place and returns it. Skips `pre_mission` and `post_mission`.

**Call before `AviaryProblem.load_inputs`.**

#### `get_default_engine()`

Returns the first configured `EngineTableBuilder`. Pass it to `load_external_subsystems` so Aviary has an engine to attach during `check_and_preprocess_inputs`; it's also used as the fallback engine in `MultiPhasePropulsionBuilder`.

#### `install_propulsion(aviary_group, propulsion_name='propulsion')`

Walks `aviary_group.subsystems` and replaces the first builder named `propulsion_name` with a `MultiPhasePropulsionBuilder` that carries this builder's per-phase engine mapping.

**Call after `check_and_preprocess_inputs` and before `build_model`.** Raises `RuntimeError` if no builder named `propulsion_name` is found.

#### `build_post_mission(...)`

Returns a `MultiEngineFuelBurnComp` that `AviaryGroup` adds to its `post_mission` group under this builder's name. Invoked automatically by Aviary during `build_model`.

#### `wire_trajectory(aviary_group)`

Connects each regular phase's `traj.{phase}.timeseries.mass` (first and last node) into the post-mission component's `mass_start_{phase}` / `mass_end_{phase}` inputs. Skips phases not in `aviary_group.regular_phases`.

**Call after `build_model` and before `setup`.**

### `MultiEngineFuelBurnComp`

OpenMDAO `ExplicitComponent` that computes per-fuel mass and volume totals from phase start/end mass inputs. Sums `mass_start - mass_end` across all phases that share the same `(csv, density)` pair, then divides mass by density to produce volume. Defined internally; instantiated by `MultiEngineTableBuilder.build_post_mission`.

## Output Variables

After the mission runs, per-fuel totals are available via the module-level names:

```python
from multi_fuel.table_builder import TOTAL_FUEL_MULTI, TOTAL_FUEL_VOLUME_MULTI

fuel_mass = prob.get_val(TOTAL_FUEL_MULTI, units='lbm')
fuel_volume = prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS')
```

These resolve to `'mission:total_fuel_multi'` and `'mission:total_fuel_volume_multi'` respectively. They are defined in `multi_fuel/table_builder.py` rather than Aviary's `Mission` namespace so the extension works against an unmodified Aviary install.

Both arrays are indexed by unique `(csv, density)` pair in order of first appearance.

**Example index mapping:**

| Index | CSV | Density (lbm/galUS) | Phases contributing |
|-------|-----|---------------------|---------------------|
| 0 | `turbofan_28k.csv` | 6.7 | climb |
| 1 | `turbofan_22k.csv` | 6.4 | cruise |
| 2 | `turbofan_22k.csv` | 3.5 | descent |

**Notes:**

- Array length equals the number of unique `(csv, density)` pairs.
- Per-phase fuel mass is `mass[t=0] - mass[t=final]` on that phase's mass timeseries.
- Volume is `mass / density`.
- Only regular trajectory phases are wired; reserve, taxi, and takeoff phases are not included.

## Aviary Integration

**No files under `aviary/` are modified.** The extension composes Aviary's public API:

- `SubsystemBuilder` (base class for `MultiEngineTableBuilder`)
- `CorePropulsionBuilder` and `PropulsionMission` (`MultiPhasePropulsionBuilder` subclasses / composes them)
- `EngineDeck` (base class for `EngineTableBuilder`)
- `AviaryProblem.load_external_subsystems`, `check_and_preprocess_inputs`, `build_model`
- `AviaryGroup.subsystems` list, `regular_phases` attribute, `connect` method

## Running the Example

```bash
pixi run python multi_fuel/run_single_aisle.py
```

The example mission uses:

| Phase | CSV | Density (lbm/galUS) |
|-------|-----|---------------------|
| climb | `turbofan_24k_1.csv` | 6.4 |
| cruise | `turbofan_28k.csv` | 6.4 |
| descent | `turbofan_22k.csv` | 6.4 |

## Verifying Per-Phase Dispatch

A quick sanity check that the per-phase engine actually drives that phase's mass dynamics: zero the `Fuel Flow (lb/h, output)` column of the CSV assigned to a single phase and re-run. Only that phase's mass delta should go to zero; the others are unaffected. The benchmark test's `_PHASE_ENGINE_MAP` exercises three distinct CSVs so the effect is isolable per phase.

## When to Use This Extension

Use `multi_fuel` when you need to:

- Simulate aircraft operations with different fuel types across mission phases (Jet-A, SAF blend, LNG, etc.)
- Model engine performance variations between different engine decks across phases
- Track fuel consumption separately by fuel type for reporting and analysis
- Optimize mission profiles with multi-fuel constraints

## Limitations

- Only CSV-based engine decks (not parameterized engine models) are supported.
- Fuel density is constant within each phase.
- Pre-mission sizing uses the default engine (Aviary's pre-mission is not phase-aware).
- Taxi and takeoff fuel burn are not included in the multi-fuel output arrays.

## See Also

- [Aviary Documentation](https://openmdao.org/aviary)
- `multi_fuel/table_builder.py` — Source with detailed docstrings
- `multi_fuel/docs/index.html` — HTML documentation with syntax-highlighted examples
- `aviary/subsystems/propulsion/propulsion_builder.py` — Base `CorePropulsionBuilder`
- `aviary/subsystems/propulsion/engine_deck.py` — Base `EngineDeck`
