"""Optimize fuel burn for a Large Single Aisle 2 aircraft with multi-fuel engine decks.

This script demonstrates a complete Aviary optimization problem for a Large Single
Aisle 2 (LSA2) aircraft configured to use different engine decks and fuel types
across different mission phases. The optimization minimizes total fuel consumption
while satisfying trajectory constraints.

The mission profile consists of three phases:
    climb : Uses turbofan_28k.csv engine deck with Jet-A fuel (6.7 lbm/galUS)
    cruise : Uses turbofan_22k.csv engine deck with SAF blend fuel (6.4 lbm/galUS)
    descent : Uses turbofan_22k.csv engine deck with LNG fuel (6.4 lbm/galUS)

The multi-fuel capability is provided by the MultiEngineTableBuilder class in the
multi_fuel.table_builder module, which manages a mapping of mission phases to
different engine decks with optional per-phase fuel densities.

Workflow
--------
1. Load LSA2 aircraft inputs and modify engine parameters
2. Configure multi-engine phase mapping with different fuel types
3. Build the AviaryProblem with IPOPT optimizer
4. Run the optimization to minimize fuel burn
5. Report total and per-engine fuel consumption

Aircraft Configuration
----------------------
    Engine Mass : 6293.8 lbm
    Reference Mass : 6293.8 lbm
    Reference SLS Thrust : 22200.5 lbf
    Scaled SLS Thrust : 22200.5 lbf
    Scale Factor : 1.0

Mission Phases
--------------
The phase_info is loaded from aviary.models.missions.energy_state_default and
modified via deepcopy to avoid mutating the original. The default mission includes
climb, cruise, and descent phases with constraints on altitude, Mach number,
and range.

Optimization
------------
The optimizer is set to IPOPT with a maximum of 50 iterations and using
coloring for efficient gradient computation. The objective is to minimize
fuel burn (mission:objectives:fuel).

Expected Output
---------------
Upon successful completion, the script prints:
    - Optimization success status (True/False)
    - Total fuel consumed in lbm
    - Per-engine fuel breakdown in lbm (by fuel type)
    - Per-engine fuel breakdown in US gallons (by fuel type)

Examples
--------
Run the script directly::

    pixi run python multi_fuel/run_single_aisle.py

To modify the optimization parameters, edit the AviaryProblem configuration
in the ``if __name__ == '__main__'`` block. To add or modify mission phases,
edit the phase_engine_map and phase_info variables.

See Also
--------
multi_fuel.table_builder.MultiEngineTableBuilder : Multi-fuel engine builder
multi_fuel.table_builder.MultiEngineFuelBurnComp : Fuel burn computation
aviary.models.aircraft.large_single_aisle_2 : LSA2 aircraft model
aviary.models.missions.energy_state_default : Default mission phase definitions
"""

from copy import deepcopy

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    inputs as lsa2_inputs,
)
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Aircraft
from multi_fuel.table_builder import MultiEngineTableBuilder

phase_info = deepcopy(phase_info)

inputs = deepcopy(lsa2_inputs)

inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

engine = MultiEngineTableBuilder(
    phase_engine_map={
        'climb': ('models/engines/turbofan_28k.csv', 6.7),
        'cruise': ('models/engines/turbofan_22k.csv', 6.4),
        'descent': ('models/engines/turbofan_22k.csv', 6.4),
    },
)
phase_info = engine.configure_phase_info(phase_info)

if __name__ == '__main__':
    """Execute the Aviary optimization problem for multi-fuel LSA2 aircraft.

    This block constructs and runs a complete AviaryProblem optimization that
    minimizes fuel burn for a Large Single Aisle 2 aircraft using different
    engine decks and fuel types across mission phases.

    The optimization workflow:
        1. Create AviaryProblem instance
        2. Load aircraft inputs and phase info
        3. Register multi-fuel engine as external subsystem
        4. Validate and preprocess inputs
        5. Build the full aircraft model
        6. Configure IPOPT optimizer with coloring
        7. Add design variables and objective
        8. Setup the problem
        9. Run the optimization
        10. Print fuel consumption results
    """
    prob = AviaryProblem()

    prob.load_inputs(inputs, phase_info)

    prob.load_external_subsystems([engine])

    prob.check_and_preprocess_inputs()

    prob.build_model()

    prob.add_driver('IPOPT', max_iter=50, use_coloring=True)

    prob.add_design_variables()
    prob.add_objective()

    prob.setup()

    prob.run_aviary_problem()

    print('Success:', prob.result.success)
    from aviary.variable_info.variables import Mission

    print('Total fuel (lbm):', prob.get_val(Mission.TOTAL_FUEL, units='lbm'))
    print(
        'Per-engine fuel (lbm):',
        prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm'),
    )
    print(
        'Per-engine fuel (galUS):',
        prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS'),
    )
