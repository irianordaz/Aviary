"""Optimize fuel burn for a Large Single Aisle 2 aircraft with multi-fuel engine decks.

This script demonstrates a complete Aviary optimization problem for a Large Single
Aisle 2 (LSA2) aircraft configured to report fuel consumption across multiple
engine decks / fuel types via a post-mission accounting subsystem built on top of
``aviary.subsystems.subsystem_builder.SubsystemBuilder``.

The mission profile consists of three phases (climb, cruise, descent). The multi-
fuel reporting is provided by ``MultiEngineTableBuilder`` which is registered as
an external SubsystemBuilder; Aviary's actual propulsion uses the engine
returned by ``get_default_engine()`` (the first entry in the phase/engine map).

Workflow
--------
1. Load LSA2 aircraft inputs and modify engine parameters
2. Configure the multi-engine phase mapping with optional per-phase fuel densities
3. Register the default engine and the builder as external subsystems
4. Build the model, wire trajectory mass timeseries, and run the optimization
5. Report total and per-engine fuel consumption
"""

from copy import deepcopy

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    inputs as lsa2_inputs,
)
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Aircraft, Mission
from multi_fuel.table_builder import (
    TOTAL_FUEL_MULTI,
    TOTAL_FUEL_VOLUME_MULTI,
    MultiEngineTableBuilder,
)

phase_info = deepcopy(phase_info)
inputs = deepcopy(lsa2_inputs)

inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

engine = MultiEngineTableBuilder(
    phase_engine_map={
        'climb': ('multi_fuel/engines/turbofan_24k_1.csv', 6.4),
        'cruise': ('multi_fuel/engines/turbofan_22k.csv', 6.4),
        'descent': ('multi_fuel/engines/turbofan_28k.csv', 6.4),
    },
)

if __name__ == '__main__':
    prob = AviaryProblem()

    # Tag each phase so MultiEngineTableBuilder.build_mission can dispatch to
    # the CSV/density configured for that phase in phase_engine_map.
    phase_info = engine.configure_phase_info(phase_info)

    prob.load_inputs(inputs, phase_info)

    # Register the default engine (an EngineDeck → EngineModel) for Aviary
    # propulsion, and the builder itself as an external SubsystemBuilder that
    # contributes the post-mission fuel accounting component.
    prob.load_external_subsystems([engine.get_default_engine(), engine])

    prob.check_and_preprocess_inputs()

    prob.build_model()

    # build_model() adds the post-mission fuel burn component under
    # post_mission.<engine.name>; connect phase mass timeseries to its inputs
    # before setup().
    engine.wire_trajectory(prob.model)

    prob.add_driver('IPOPT', max_iter=50, use_coloring=True)

    prob.add_design_variables()
    prob.add_objective()

    prob.setup()

    prob.run_aviary_problem()

    print('\nSuccess:', not prob.result)
    print('Total fuel (lbm):', prob.get_val(Mission.TOTAL_FUEL, units='lbm'))
    print(
        'Per-engine fuel (lbm):',
        prob.get_val(TOTAL_FUEL_MULTI, units='lbm'),
    )
    print(
        'Per-engine fuel (galUS):',
        prob.get_val(TOTAL_FUEL_VOLUME_MULTI, units='galUS'),
    )
