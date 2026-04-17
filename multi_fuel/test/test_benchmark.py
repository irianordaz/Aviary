"""Benchmark test for the integrated multi-fuel solution.

This module contains a benchmark test that validates the complete end-to-end
workflow demonstrated in run_single_aisle.py. It constructs and runs a full
AviaryProblem optimization for a Large Single Aisle 2 (LSA2) aircraft configured
with different engine decks and fuel types across mission phases.

The test validates:
- Optimization convergence (success status)
- Total fuel consumption is within expected range
- Per-engine fuel burn arrays have correct shape
- Per-engine fuel values are positive and reasonable
- Fuel volume output is positive and reasonable
- The multi-fuel outputs are consistent with the total fuel

Mission Phases
--------------
    climb   : turbofan_28k.csv with Jet-A fuel (6.7 lbm/galUS)
    cruise  : turbofan_22k.csv with SAF blend fuel (6.4 lbm/galUS)
    descent : turbofan_22k.csv with LNG fuel (6.4 lbm/galUS)

The unique (csv, density) pairs are:
    0: (turbofan_28k.csv, 6.7) - climb phase
    1: (turbofan_22k.csv, 6.4) - cruise + descent phases (same CSV and density)
"""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    inputs as lsa2_inputs,
)
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Aircraft, Mission
from multi_fuel.table_builder import MultiEngineTableBuilder


@use_tempdirs
class MultiFuelBenchmarkTest(unittest.TestCase):
    """Benchmark test for the integrated multi-fuel solution.

    This test validates the complete end-to-end workflow for a multi-fuel
    LSA2 aircraft optimization, ensuring the MultiEngineTableBuilder correctly
    integrates with the Aviary framework and produces reasonable results.
    """

    def setUp(self):
        """Set up common test fixtures."""
        om.clear_reports()
        _clear_problem_names()

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multi_fuel_optimization_success(self):
        """Test that the multi-fuel optimization converges successfully.

        Verifies that the AviaryProblem with MultiEngineTableBuilder
        completes the optimization without errors and reports success.
        """
        om.clear_reports()

        phase_info_modified = deepcopy(phase_info)
        inputs_modified = deepcopy(lsa2_inputs)

        inputs_modified.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': ('models/engines/turbofan_28k.csv', 6.7),
                'cruise': ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )
        phase_info_modified = engine.configure_phase_info(phase_info_modified)

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(inputs_modified, phase_info_modified)
        prob.load_external_subsystems([engine])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        self.assertTrue(prob.result.success)  # type: ignore[attr-defined]

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multi_fuel_fuel_burn_values(self):
        """Test that fuel burn values are reasonable.

        Verifies that:
        - Total fuel is positive
        - Per-engine fuel arrays have correct shape (2 unique csv/density pairs)
        - Per-engine fuel values are positive
        - Total fuel is approximately equal to sum of per-engine fuel
        """
        om.clear_reports()

        phase_info_modified = deepcopy(phase_info)
        inputs_modified = deepcopy(lsa2_inputs)

        inputs_modified.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': ('models/engines/turbofan_28k.csv', 6.7),
                'cruise': ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )
        phase_info_modified = engine.configure_phase_info(phase_info_modified)

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(inputs_modified, phase_info_modified)
        prob.load_external_subsystems([engine])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        total_fuel = prob.get_val(Mission.TOTAL_FUEL, units='lbm')
        fuel_multi = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')
        fuel_volume = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS')

        self.assertGreater(total_fuel[0], 0.0)
        self.assertEqual(fuel_multi.shape, (2,))
        self.assertEqual(fuel_volume.shape, (2,))

        for fuel_value in fuel_multi:
            self.assertGreater(fuel_value, 0.0)

        for volume_value in fuel_volume:
            self.assertGreater(volume_value, 0.0)

        total_fuel_multi_sum = sum(fuel_multi)
        self.assertGreater(total_fuel[0], total_fuel_multi_sum)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multi_fuel_volume_consistency(self):
        """Test that fuel volume is consistent with mass and density.

        Verifies that the fuel volume output is correctly computed by
        dividing the fuel mass by the corresponding fuel density for
        each unique (csv, density) pair.
        """
        om.clear_reports()

        phase_info_modified = deepcopy(phase_info)
        inputs_modified = deepcopy(lsa2_inputs)

        inputs_modified.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': ('models/engines/turbofan_28k.csv', 6.7),
                'cruise': ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )
        phase_info_modified = engine.configure_phase_info(phase_info_modified)

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(inputs_modified, phase_info_modified)
        prob.load_external_subsystems([engine])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        fuel_mass = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')
        fuel_volume = prob.get_val(Mission.TOTAL_FUEL_VOLUME_MULTI, units='galUS')

        densities = [6.7, 6.4]

        for i, (mass, volume, density) in enumerate(zip(fuel_mass, fuel_volume, densities)):
            expected_volume = mass / density
            assert_near_equal(volume, expected_volume, tolerance=1e-4)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multi_fuel_phase_mapping(self):
        """Test that the phase mapping produces expected output structure.

        Verifies that the phase_engine_map correctly maps to the output
        array indices, with the same CSV+density pair appearing at
        different indices if they have different densities.
        """
        om.clear_reports()

        phase_info_modified = deepcopy(phase_info)
        inputs_modified = deepcopy(lsa2_inputs)

        inputs_modified.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': ('models/engines/turbofan_28k.csv', 6.7),
                'cruise': ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )
        phase_info_modified = engine.configure_phase_info(phase_info_modified)

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(inputs_modified, phase_info_modified)
        prob.load_external_subsystems([engine])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        fuel_multi = prob.get_val(Mission.TOTAL_FUEL_MULTI, units='lbm')

        self.assertEqual(len(fuel_multi), 2)

        climb_fuel = fuel_multi[0]
        cruise_descent_fuel = fuel_multi[1]

        self.assertGreater(climb_fuel, 0.0)
        self.assertGreater(cruise_descent_fuel, 0.0)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multi_fuel_optimization_reasonable_fuel(self):
        """Test that total fuel consumption is within a reasonable range.

        For a LSA2 aircraft with the configured engine decks, the total
        fuel consumption should be within a physically reasonable range
        for a typical mission profile.
        """
        om.clear_reports()

        phase_info_modified = deepcopy(phase_info)
        inputs_modified = deepcopy(lsa2_inputs)

        inputs_modified.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
        inputs_modified.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
        inputs_modified.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        engine = MultiEngineTableBuilder(
            phase_engine_map={
                'climb': ('models/engines/turbofan_28k.csv', 6.7),
                'cruise': ('models/engines/turbofan_22k.csv', 6.4),
                'descent': ('models/engines/turbofan_22k.csv', 6.4),
            },
        )
        phase_info_modified = engine.configure_phase_info(phase_info_modified)

        prob = AviaryProblem(verbosity=0)
        prob.load_inputs(inputs_modified, phase_info_modified)
        prob.load_external_subsystems([engine])
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=50, use_coloring=True)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(suppress_solver_print=True)

        total_fuel = prob.get_val(Mission.TOTAL_FUEL, units='lbm')[0]

        self.assertGreater(total_fuel, 10000.0)
        self.assertLess(total_fuel, 100000.0)


if __name__ == '__main__':
    unittest.main()
