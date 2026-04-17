"""Aviary problem: Large Single Aisle 2 aircraft with turbofan_22k.csv engine deck."""

from copy import deepcopy

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    inputs as lsa2_inputs,
)
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Aircraft
from dual_fuel.table_builder import EngineTableBuilder

phase_info = deepcopy(phase_info)

inputs = deepcopy(lsa2_inputs)

inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

engine1 = EngineTableBuilder(csv_path='models/engines/turbofan_22k.csv')
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')

# engine2 = EngineTableBuilder(csv_path='aviary/models/engines/turbofan_28k.csv')
# inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 28928.1, 'lbf')
# inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 28928.1, 'lbf')

if __name__ == '__main__':
    prob = AviaryProblem()

    prob.load_inputs(inputs, phase_info)

    prob.load_external_subsystems([engine1])

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
